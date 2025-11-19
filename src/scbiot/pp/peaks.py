from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import pyranges as pr
import re
import scipy.sparse as sp
from anndata import AnnData
from numpy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.utils.sparsefuncs import inplace_column_scale
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
import scanpy as sc


# ---------------------------
# Helpers added (techniques 1, 4, 5)
# ---------------------------
_GTF_CACHE: dict[str, pr.PyRanges] = {}


def ensure_csr_f32(X: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr(copy=False).astype(np.float32, copy=False)
    return sp.csr_matrix(np.asarray(X, dtype=np.float32, order="C"))


def _clean_version(s: str) -> str:
    s = str(s)
    if "." in s and s.split(".")[-1].isdigit():
        s = s[: s.rfind(".")]
    return s


def _lsi_fit_project(
    X_tfidf: sp.csr_matrix,
    n_components: int,
    sample_cells_pre: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, TruncatedSVD]:
    """Technique 1: fit SVD on a subset of cells, then project all cells."""
    n = X_tfidf.shape[0]
    if sample_cells_pre is not None and sample_cells_pre > 0 and sample_cells_pre < n:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, sample_cells_pre, replace=False)
        X_fit = X_tfidf[idx]
    else:
        X_fit = X_tfidf

    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=7,
        random_state=random_state,
        algorithm="randomized",
    )
    svd.fit(X_fit)
    X_lsi = svd.transform(X_tfidf)  # project ALL cells
    return X_lsi, svd


def _winsorize_tfidf_inplace(tfidf: sp.csr_matrix, q: Tuple[float, float]) -> sp.csr_matrix:
    """Technique 5: winsorize (clip) per-row at given quantiles (in-place on data)."""
    if not sp.isspmatrix_csr(tfidf):
        tfidf = tfidf.tocsr()
    lo, hi = q
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("outlier_quantiles must satisfy 0 <= lo < hi <= 1")
    indptr = tfidf.indptr
    data = tfidf.data
    for i in range(tfidf.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        if e > s:
            v = data[s:e]
            a, b = np.quantile(v, [lo, hi])
            if a == b:
                continue
            v[v < a] = a
            v[v > b] = b
    return tfidf


def _select_features_per_cluster_union(
    X_counts: sp.csr_matrix,
    clusters: np.ndarray,
    topN: int,
    per_cluster_top: Optional[int] = None,
) -> np.ndarray:
    """Technique 4: union of per-cluster top features by accessibility (binary sums), capped at topN."""
    X_counts = X_counts.tocsr() if not sp.isspmatrix_csr(X_counts) else X_counts
    Xbin = (X_counts > 0).astype(np.int8)

    labs = np.unique(clusters)
    n_clusters = len(labs)
    if per_cluster_top is None:
        per_cluster_top = max(1, topN // max(1, n_clusters))

    sel = set()
    global_score = np.asarray(Xbin.sum(axis=0)).ravel()

    for lab in labs:
        rows = (clusters == lab)
        if rows.sum() == 0:
            continue
        score = np.asarray(Xbin[rows].sum(axis=0)).ravel()
        topk = np.argsort(score)[::-1][:per_cluster_top]
        topk = topk[score[topk] > 0]
        sel.update(topk.tolist())
        if len(sel) >= topN:
            break

    idx = np.array(sorted(sel))
    if idx.size > topN:
        g = global_score[idx]
        keep = np.argsort(g)[::-1][:topN]
        idx = idx[keep]
    return idx


# ---------------------------
# Your original functions
# ---------------------------

def remove_promoter_proximal_peaks(
    adata: AnnData,
    gtf_file: Path | str,
    promoter_up: int = 2000,
    promoter_down: int = 500,
    chrom_col: Optional[str] = None,
    start_col: Optional[str] = None,
    end_col: Optional[str] = None,
) -> AnnData:
    var = adata.var
    if chrom_col and start_col and end_col and {chrom_col, start_col, end_col}.issubset(var.columns):
        peaks_df = var[[chrom_col, start_col, end_col]].rename(
            columns={chrom_col: "Chromosome", start_col: "Start", end_col: "End"}
        )
        peaks_df["peak_name"] = adata.var_names.astype(str)
    elif {"chrom", "chromStart", "chromEnd"}.issubset(var.columns):
        peaks_df = var[["chrom", "chromStart", "chromEnd"]].rename(
            columns={"chrom": "Chromosome", "chromStart": "Start", "chromEnd": "End"}
        )
        peaks_df["peak_name"] = adata.var_names.astype(str)
    else:
        patterns = [
            re.compile(r"^(?P<Chromosome>[^:_-]+)[:_](?P<Start>\d+)[-_](?P<End>\d+)$"),
            re.compile(r"^(?P<Chromosome>[^:_-]+)[-_](?P<Start>\d+)[-_](?P<End>\d+)$"),
        ]
        rows, bad = [], []
        for peak in map(str, adata.var_names.tolist()):
            for patt in patterns:
                match = patt.match(peak)
                if match:
                    d = match.groupdict()
                    rows.append((d["Chromosome"], int(d["Start"]), int(d["End"]), peak))
                    break
            else:
                bad.append(peak)
        if bad:
            raise ValueError(
                f"Could not parse {len(bad)} peak names (first few: {bad[:5]}). "
                "Expect chr1_100_200, chr1-100-200, or chr1:100-200 formats."
            )
        peaks_df = pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "peak_name"])

    peaks_df["Start"] = peaks_df["Start"].astype(int)
    peaks_df["End"] = peaks_df["End"].astype(int)
    peaks_df["Chromosome"] = peaks_df["Chromosome"].astype(str)
    peaks = pr.PyRanges(peaks_df)

    genes = pr.read_gtf(str(gtf_file))
    genes = genes[genes.Feature == "gene"]

    def _harmonize_chr(peaks_gr: pr.PyRanges, genes_gr: pr.PyRanges) -> pr.PyRanges:
        p_has_chr = peaks_gr.df["Chromosome"].astype(str).str.startswith("chr").any()
        g_has_chr = genes_gr.df["Chromosome"].astype(str).str.startswith("chr").any()
        gdf = genes_gr.df.copy()
        if p_has_chr and not g_has_chr:
            gdf["Chromosome"] = "chr" + gdf["Chromosome"].astype(str)
        elif not p_has_chr and g_has_chr:
            gdf["Chromosome"] = gdf["Chromosome"].astype(str).str.replace(r"^chr", "", regex=True)
        return pr.PyRanges(gdf)

    genes = _harmonize_chr(peaks, genes)

    promoters = []
    for row in genes.df.itertuples(index=False):
        if row.Strand == "+":
            start = max(0, row.Start - promoter_up)
            end = row.Start + promoter_down
        else:
            start = max(0, row.End - promoter_down)
            end = row.End + promoter_up
        promoters.append([row.Chromosome, start, end])
    promoters_df = pd.DataFrame(promoters, columns=["Chromosome", "Start", "End"])
    promoters = pr.PyRanges(promoters_df)

    overlap = peaks.join(promoters)
    overlap_names = set(overlap.df["peak_name"].astype(str))
    adata.var["is_promoter_proximal"] = adata.var_names.astype(str).isin(overlap_names)

    kept = [name for name in adata.var_names if name not in overlap_names]
    n_removed = adata.n_vars - len(kept)
    if n_removed > 0:
        print(
            f"Removed {n_removed:,} promoter-proximal peaks "
            f"({promoter_up}bp upstream / {promoter_down}bp downstream). "
            f"Remaining: {len(kept):,}"
        )
    return adata[:, kept].copy()


def find_variable_features(
    adata: AnnData,
    batch_key: str,
    layer: str = "counts",
    per_batch_min_frac: float = 0.01,
    topN: int = 30000,
    set_var_flag: Optional[str] = None,
    normalize_output: bool = True,
    idf_alpha: float = 1.0,
    hetero_gamma: float = 2.0,
    layer_out: str = "X",
    add_key: str = "var_features",
) -> AnnData:
    X = adata.layers[layer] if layer in adata.layers else adata.X
    X = X if sp.issparse(X) else sp.csr_matrix(X)

    n_cells, n_feats = X.shape
    batch_vals = adata.obs[batch_key].values
    batch_cats = np.unique(batch_vals)

    common_mask = np.ones(n_feats, dtype=bool)
    batch_sizes: List[int] = []
    batch_rows_list: List[np.ndarray] = []
    for b in batch_cats:
        rows = batch_vals == b
        batch_rows_list.append(rows)
        n_b = int(rows.sum())
        batch_sizes.append(n_b)
        if n_b > 0:
            nnz_frac = X[rows].getnnz(axis=0) / n_b
            common_mask &= nnz_frac >= per_batch_min_frac

    keep_common = np.flatnonzero(common_mask)
    if keep_common.size == 0:
        raise ValueError("No features pass per-batch commonness filter.")

    X_sub = X[:, keep_common].tocsr()
    Xbin = (X_sub > 0).astype(np.int8)
    n_keep = X_sub.shape[1]
    batch_sizes = np.array(batch_sizes, dtype=float)
    total_cells = batch_sizes.sum()

    P = np.zeros((len(batch_cats), n_keep), dtype=np.float32)
    Var_in = np.zeros_like(P)
    for i, rows in enumerate(batch_rows_list):
        n_b = batch_sizes[i]
        if n_b > 0:
            p_b = np.asarray(Xbin[rows].mean(axis=0)).ravel()
            P[i] = p_b
            Var_in[i] = p_b * (1 - p_b)

    w = (batch_sizes / total_cells)[:, None]
    within_score = (w * Var_in).sum(axis=0)
    mean_p = (w * P).sum(axis=0)
    var_between = (w * (P - mean_p) ** 2).sum(axis=0)
    hetero_penalty = 1.0 / (1.0 + hetero_gamma * var_between)

    df = np.asarray(Xbin.getnnz(axis=0)).ravel()
    idf = np.log1p(total_cells / (1.0 + df))
    idf_factor = idf ** idf_alpha if idf_alpha != 0 else np.ones_like(idf)

    var_scores = within_score * hetero_penalty * idf_factor
    mean_detected = Xbin.mean(axis=0).A1 if sp.issparse(Xbin) else Xbin.mean(axis=0)
    ubiquitous_mask = mean_detected > 0.9
    var_scores[ubiquitous_mask] *= 0.5

    sel_idx = keep_common[np.argsort(var_scores)[::-1][:min(topN, n_keep)]]

    if set_var_flag is not None:
        flag = np.zeros(n_feats, dtype=bool)
        flag[sel_idx] = True
        adata.var[set_var_flag] = flag

    X_sel = X[:, sel_idx].astype(np.float32)
    if normalize_output:
        lib_size = np.asarray(X_sel.sum(axis=1)).ravel()
        lib_size[lib_size == 0] = 1.0
        if sp.issparse(X_sel):
            X_sel = sp.diags(1.0 / lib_size).dot(X_sel)
            mean_sq = np.asarray(X_sel.power(2).mean(axis=0)).ravel()
            std = np.sqrt(np.maximum(mean_sq, 1e-8))
            X_sel = X_sel.multiply(1 / std)
        else:
            X_sel = X_sel / lib_size[:, None]
            std = np.sqrt(np.maximum(X_sel.var(axis=0), 1e-8))
            X_sel = X_sel / std

    X_sel = X_sel.tocsr() if sp.issparse(X_sel) else np.asarray(X_sel, dtype=np.float32)

    adata.uns[add_key] = AnnData(
        X=X_sel if layer_out == "X" else None,
        obs=adata.obs.copy(),
        var=adata.var.iloc[sel_idx].copy(),
        layers={layer_out: X_sel} if layer_out != "X" else {},
    )
    return adata.uns[add_key]


def tfidf_transform(
    adata: AnnData,
    scale_factor: float = 1.0,   # <- default to 1.0 to avoid extra log scaling
    layer_out: str = "tfidf",
    use_float32: bool = True,
) -> sp.csr_matrix:
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    X = X.astype(np.float32 if use_float32 else X.dtype, copy=False).tocsr()

    # Binarize counts first (canonical scATAC TF-IDF)
    X = (X > 0).astype(np.float32).tocsr()

    tf = X.copy()
    inplace_csr_row_normalize_l1(tf)
    if scale_factor != 1.0:
        tf.data = np.log1p(tf.data * scale_factor + 1e-6)

    N = X.shape[0]
    df = np.asarray(X.getnnz(axis=0)).ravel()
    idf = np.log(1.0 + N / (1.0 + df))

    tfidf = tf.copy()
    inplace_column_scale(tfidf, idf)
    adata.layers[layer_out] = tfidf
    return tfidf


def lsi_transform(
    adata: AnnData,
    n_components: int = 31,
    n_iter: int = 5,
    topN: int = 30000,
    layer: str = "counts",
    drop_first_component: bool = True,
    random_state: int = 42,
    cluster_k: int = 10,
    normalize_output: bool = True,
    add_key: str = "X_lsi",
    # ---- Existing kwargs preserved ----
    sample_cells_pre: Optional[int] = 10000,             # Technique 1 (fit on subset)
    per_cluster_union: bool = True,                    # Technique 4 (off by default)
    per_cluster_top: Optional[int] = 4000,              # Technique 4 control
    outlier_quantiles: Optional[Tuple[float, float]] = (0.02, 0.98),  # Technique 5 (e.g., (0.02, 0.98))
) -> np.ndarray:
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if not sp.isspmatrix_csr(X):
        X = sp.csr_matrix(X)
    X = X.astype(np.float32)

    # Binary view used for frequency-based feature stats
    Xbin_full = (X > 0).astype(np.int8)

    features_idx = np.arange(X.shape[1])
    X_use = X

    for j in range(1, n_iter + 1):
        print(f"Running Iterative LSI iteration {j} ...")

        # --- TF on current feature subset ---
        tf = X_use.copy()
        inplace_csr_row_normalize_l1(tf)

        # Optional winsorization on TF (clip before IDF)
        if outlier_quantiles is not None:
            _winsorize_tfidf_inplace(tf, outlier_quantiles)

        # --- IDF from binary df on current subset ---
        df = np.asarray((X_use > 0).getnnz(axis=0)).ravel()
        idf = np.log(1.0 + X_use.shape[0] / (1.0 + df))

        tfidf = tf.copy()
        inplace_column_scale(tfidf, idf)

        # Technique 1: fit SVD on subset, project all
        X_lsi, svd = _lsi_fit_project(
            X_tfidf=tfidf,
            n_components=n_components,
            sample_cells_pre=sample_cells_pre,
            random_state=random_state,
        )

        if normalize_output:
            X_lsi = sk_normalize(X_lsi, norm="l2", copy=False)
        if drop_first_component and X_lsi.shape[1] > 1:
            X_lsi = X_lsi[:, 1:]

        adata.obsm[f"{add_key}_iter{j}"] = X_lsi

        if j == n_iter:
            break

        # --- Clustering on current embedding ---
        km = KMeans(
            n_clusters=max(2, min(cluster_k, X_lsi.shape[0] // 50)),
            n_init=10,                  # portable across sklearn versions
            random_state=random_state
        )
        clusters = km.fit_predict(X_lsi)

        # --- Feature selection on binary frequencies ---
        if per_cluster_union:
            features_idx = _select_features_per_cluster_union(
                X_counts=X,  # helper binarizes internally
                clusters=clusters,
                topN=min(topN, X.shape[1]),
                per_cluster_top=per_cluster_top,
            )
        else:
            uniq = np.unique(clusters)
            cluster_means = []
            for lab in uniq:
                rows = (clusters == lab)
                cluster_means.append(np.asarray(Xbin_full[rows].mean(axis=0)).ravel())
            cluster_means = np.vstack(cluster_means)
            feature_var = cluster_means.var(axis=0)

            # fast top-N without full sort
            k = min(topN, X.shape[1])
            idx = np.argpartition(feature_var, -k)[-k:]
            features_idx = idx[np.argsort(feature_var[idx])[::-1]]

        # Subset features for next iteration
        X_use = X[:, features_idx]

    adata.obsm[add_key] = adata.obsm[f"{add_key}_iter{n_iter}"]
    return adata.obsm[add_key]


def add_iterative_lsi(
    adata: AnnData,
    n_components: int = 51,
    drop_first_component: bool = True,
    tfidf_layer: str = "tfidf",
    add_key: str = "X_lsi",
    **lsi_kwargs: Any,
) -> np.ndarray:
    """
    Convenience wrapper that runs iterative LSI.
    Note: We no longer precompute and pass a TF-IDF layer here to avoid double-normalization.
    """
    # Allow callers to override the layer via kwargs without clashing with the fixed default
    layer = lsi_kwargs.pop("layer", "counts")
    return lsi_transform(
        adata=adata,
        layer=layer,
        n_components=n_components,
        drop_first_component=drop_first_component,
        add_key=add_key,
        **lsi_kwargs,
    )



def annotate_gene_activity(
    atac: AnnData,
    gtf_file: Path | str,
    *,
    peak_chrom_col: str = "chrom",
    peak_start_col: str = "chromStart",
    peak_end_col: str = "chromEnd",
    gene_biotypes: tuple[str, ...] = ("protein_coding", "lncRNA"),
    promoter_up: int = 2000,
    promoter_down: int = 0,
    include_gene_body: bool = False,
    weight_by_distance: bool = True,
    tss_decay_bp: int = 2000,
    prefer_gene_name: bool = True,
    promoter_priority: bool = True,
    verbose: bool = True,
) -> AnnData:
    var = atac.var

    def _normalize_chr(series: pd.Series) -> pd.Series:
        series = series.astype(str)
        series = series.str.replace(r"^chr", "", regex=True)
        series = series.str.replace(r"^(MT|Mt|mt|M|chrM)$", "M", regex=True)
        return series

    def _peaks_from_cols(chrom_col: str, start_col: str, end_col: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Chromosome": _normalize_chr(var[chrom_col]),
                "Start": var[start_col].astype(int).clip(lower=0).values,
                "End": var[end_col].astype(int).values,
                "peak_idx": np.arange(var.shape[0], dtype=np.int64),
            }
        )

    def _peaks_from_names() -> pd.DataFrame:
        patterns = [
            re.compile(r"^(?P<Chromosome>[^:_-]+)[:_](?P<Start>\d+)[-_](?P<End>\d+)$"),
            re.compile(r"^(?P<Chromosome>[^:_-]+)[-_](?P<Start>\d+)[-_](?P<End>\d+)$"),
        ]
        rows, bad = [], []
        for i, peak in enumerate(map(str, atac.var_names.tolist())):
            for patt in patterns:
                m = patt.match(peak)
                if m:
                    d = m.groupdict()
                    rows.append(
                        (
                            d["Chromosome"],
                            int(d["Start"]),
                            int(d["End"]),
                            i,
                        )
                    )
                    break
            else:
                bad.append(peak)
        if bad:
            raise ValueError(
                f"Could not parse {len(bad)} peak names (first few: {bad[:5]}). "
                "Add chromosome/start/end columns or use chr1_100_200 style names."
            )
        return pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "peak_idx"])

    peaks_df: pd.DataFrame | None = None
    if {peak_chrom_col, peak_start_col, peak_end_col}.issubset(var.columns):
        peaks_df = _peaks_from_cols(peak_chrom_col, peak_start_col, peak_end_col)
    else:
        alternatives = [
            ("chrom", "chromStart", "chromEnd"),
            ("Chromosome", "Start", "End"),
            ("chr", "start", "end"),
        ]
        for c, s, e in alternatives:
            if {c, s, e}.issubset(var.columns):
                peaks_df = _peaks_from_cols(c, s, e)
                break

    if peaks_df is None:
        peaks_df = _peaks_from_names()
    else:
        missing = {peak_chrom_col, peak_start_col, peak_end_col} - set(var.columns)
        if missing and verbose:
            print(f"[GA] Using fallback peak columns (missing {sorted(missing)})")

    peaks_df["Start"] = peaks_df["Start"].astype(int)
    peaks_df["End"] = peaks_df["End"].astype(int)
    peaks_df["Chromosome"] = _normalize_chr(peaks_df["Chromosome"])
    if not np.all(peaks_df["End"].values >= peaks_df["Start"].values):
        raise ValueError("Some peaks have End < Start.")

    gtf_key = str(gtf_file)
    gtf = _GTF_CACHE.get(gtf_key)
    if gtf is None:
        gtf = pr.read_gtf(gtf_key)
        _GTF_CACHE[gtf_key] = gtf
    genes_pr = gtf[gtf.Feature == "gene"]
    genes_df = genes_pr.df.copy()
    if genes_df.empty:
        raise RuntimeError("No 'gene' features in the supplied GTF.")
    genes_df["Chromosome"] = _normalize_chr(genes_df["Chromosome"])

    biotype_col = "gene_biotype" if "gene_biotype" in genes_df.columns else (
        "gene_type" if "gene_type" in genes_df.columns else None
    )
    if biotype_col is not None:
        before = len(genes_df)
        genes_df = genes_df[genes_df[biotype_col].isin(gene_biotypes)]
        if verbose:
            print(f"[GA] Kept {len(genes_df):,}/{before:,} genes by biotype {list(gene_biotypes)}")
    if genes_df.empty:
        raise RuntimeError("No genes remain after biotype filtering.")

    peaks_contigs = pd.Index(peaks_df["Chromosome"].unique())
    genes_contigs = pd.Index(genes_df["Chromosome"].unique())
    common = peaks_contigs.intersection(genes_contigs)
    if verbose:
        print(f"[GA] Peaks contigs: {len(peaks_contigs)}; Genes contigs: {len(genes_contigs)}; Common: {len(common)}")
    if len(common) == 0:
        raise RuntimeError("No chromosome names in common after normalization.")

    peaks_df = peaks_df[peaks_df["Chromosome"].isin(common)].copy()
    genes_df = genes_df[genes_df["Chromosome"].isin(common)].copy()

    gene_name_col = "gene_name" if prefer_gene_name and "gene_name" in genes_df.columns else "gene_id"
    if verbose:
        print(f"[GA] Using gene field: {gene_name_col}")

    plus_strand = genes_df["Strand"].astype(str) == "+"
    tss = np.where(plus_strand.values, genes_df["Start"].values, genes_df["End"].values)

    prom_start = np.where(plus_strand.values, tss - promoter_up, tss - promoter_down)
    prom_end = np.where(plus_strand.values, tss + promoter_down, tss + promoter_up)
    promoter_df = pd.DataFrame(
        {
            "Chromosome": genes_df["Chromosome"].values,
            "Start": np.maximum(0, prom_start).astype(int),
            "End": np.maximum(prom_end, prom_start).astype(int),
            "gene_id": genes_df["gene_id"].values,
            "gene_name": genes_df[gene_name_col].values,
            "Strand": genes_df["Strand"].values,
            "TSS": tss,
            "region": "promoter",
        }
    )
    regions_df = promoter_df
    if include_gene_body:
        gene_body_df = pd.DataFrame(
            {
                "Chromosome": genes_df["Chromosome"].values,
                "Start": genes_df["Start"].astype(int).values,
                "End": genes_df["End"].astype(int).values,
                "gene_id": genes_df["gene_id"].values,
                "gene_name": genes_df[gene_name_col].values,
                "Strand": genes_df["Strand"].values,
                "TSS": tss,
                "region": "body",
            }
        )
        regions_df = pd.concat([promoter_df, gene_body_df], ignore_index=True)

    peaks_pr = pr.PyRanges(peaks_df.rename(columns={"peak_idx": "peak"}))
    regions_pr = pr.PyRanges(regions_df)
    overlaps = peaks_pr.join(regions_pr, how="left").df.dropna(subset=["gene_name"]).copy()
    if overlaps.empty:
        raise RuntimeError("No peak overlaps with promoter/gene annotations.")

    overlaps["dist2tss"] = np.abs(overlaps["TSS"] - overlaps["Start"])
    if promoter_priority:
        overlaps.sort_values(by=["region", "dist2tss"], inplace=True)
        overlaps = overlaps.drop_duplicates(subset=["peak", "gene_id"], keep="first")

    weights = (
        np.exp(-overlaps["dist2tss"].to_numpy(dtype=np.float32) / float(tss_decay_bp))
        if weight_by_distance
        else np.ones(len(overlaps), dtype=np.float32)
    )

    gene_labels = overlaps["gene_name"].astype(str).values
    unique_genes, inverse = np.unique(gene_labels, return_inverse=True)
    gene_index = pd.Index(unique_genes, name="gene")

    peak_idx = overlaps["peak"].astype(np.int64).to_numpy()
    gene_idx = inverse.astype(np.int64)
    peak_to_gene = sp.coo_matrix((weights, (peak_idx, gene_idx)), shape=(var.shape[0], gene_index.size)).tocsr()

    counts = ensure_csr_f32(atac.layers["counts"] if "counts" in atac.layers else atac.X)
    ga_counts = counts @ peak_to_gene

    adata_ga = AnnData(X=ga_counts, obs=atac.obs.copy(), var=pd.DataFrame(index=gene_index))
    adata_ga.var["n_peaks"] = np.asarray(peak_to_gene.sum(axis=0)).ravel().astype(np.int32)
    adata_ga.uns["provenance"] = {
        "promoter_up": int(promoter_up),
        "promoter_down": int(promoter_down),
        "include_gene_body": bool(include_gene_body),
        "weight_by_distance": bool(weight_by_distance),
        "tss_decay_bp": int(tss_decay_bp),
        "gtf_file": str(gtf_file),
        "contigs_common": int(len(common)),
        "gene_col": gene_name_col,
        "promoter_priority": bool(promoter_priority),
    }
    if verbose:
        print(f"[GA] Built GA with shape {ga_counts.shape} (cells × genes) from {var.shape[0]:,} peaks.")
    return adata_ga


def harmonize_gene_names(
    adata_rna: AnnData,
    ad_ga: AnnData,
    *,
    rna_name_col: str = "gene_name",
    verbose: bool = True,
) -> None:
    """Match GA symbols to RNA casing; strip Ensembl-like .version suffixes."""
    if rna_name_col in adata_rna.var:
        rna_names = pd.Index([_clean_version(x) for x in adata_rna.var[rna_name_col].astype(str)])
    else:
        rna_names = pd.Index([_clean_version(x) for x in adata_rna.var_names.astype(str)])
    ga_names = pd.Index([_clean_version(x) for x in ad_ga.var_names.astype(str)])

    rna_upper = rna_names.str.upper()
    ga_upper = ga_names.str.upper()
    case_map = dict(zip(rna_upper, rna_names))
    ad_ga.var_names = pd.Index([case_map.get(up, orig) for up, orig in zip(ga_upper, ad_ga.var_names)])

    if "gene_name_orig" not in adata_rna.var:
        adata_rna.var["gene_name_orig"] = adata_rna.var_names
    adata_rna.var_names = rna_names
    adata_rna.var_names_make_unique()
    ad_ga.var_names_make_unique()

    if verbose:
        overlap = len(rna_upper.intersection(ga_upper))
        print(f"[names] Harmonized symbols; overlaps (case-insensitive): {overlap:,}")


def knn_smooth_ga_on_atac(atac: AnnData, ad_ga: AnnData, *, n_neighbors: int = 50, use_rep: str = "X_lsi") -> AnnData:
    if use_rep not in atac.obsm:
        raise KeyError(f"Missing embedding '{use_rep}' in ATAC AnnData; run build_atac_lsi first.")

    sc.pp.neighbors(atac, use_rep=use_rep, n_neighbors=n_neighbors, metric="cosine")
    W = atac.obsp["connectivities"].tocsr()
    row_sum = np.asarray(W.sum(1)).ravel()
    row_sum[row_sum == 0] = 1.0
    W = W.multiply(1.0 / row_sum[:, None])

    base = ensure_csr_f32(ad_ga.layers["ga"] if "ga" in ad_ga.layers else ad_ga.X)
    ad_ga.layers["ga_smooth"] = (W @ base).astype(np.float32)
    return ad_ga