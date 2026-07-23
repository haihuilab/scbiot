#!/usr/bin/env python3
"""Reproduce the Fig. 2 label-transfer benchmark across seven datasets.

Runs supBIOT, an unintegrated-PCA baseline, Symphony, scANVI, scPoli and Seurat
on Pancreas, PBMCs, Tabula Muris Senis, Tabula Sapiens, and the Xenium COAD, HCC
and OV panels. It writes:

  results/<dataset>_<method>.json      per-run metrics
  results/<dataset>_<method>.pred.csv  per-cell predictions
  fig2_label_transfer_source_data.csv  the assembled table

Run from the repository root:

  python tutorials/benchmarks/label_transfer/reproduce_fig2_label_transfer.py --all

Frozen inputs are downloaded from Figshare article 30671669 when absent:
https://figshare.com/articles/dataset/Anndata_for_scBIOT_analysis/30671669

Tabula Sapiens is too large to host there. Download
tabula_sapiens_v2_slim_with_decontXcounts.h5ad from the Tabula Sapiens v2
collection at https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5
and rebuild it locally with --rebuild Tabula_Sapiens.

Design
------
Preprocessing happens once per dataset in `freeze`, which writes
`datasets/<name>.h5ad` plus a manifest of digests. Every method then reads that
frozen file, so cells, features, reference/query split, labels and seed are
identical across methods by construction rather than by convention. Scoring goes
through `score` and nothing else, and the evaluation set is defined by the data
(query split with a ground-truth label), never by a method's output: a method
that returns no prediction for an evaluated cell is counted wrong rather than
having that cell dropped.

Environments
------------
The methods pin mutually incompatible library versions, so this script
re-invokes itself with the interpreter each method needs (see INTERPRETERS).
Seurat runs through Rscript via the companion run_seurat.R. Set the
SCBIOT_SCANVI_PYTHON, SCBIOT_SCPOLI_PYTHON, or SCBIOT_SYMPHONY_PYTHON
environment variable to use environments outside this directory; a method
whose interpreter is missing is skipped with a note.

Required packages: the local scbiot package plus anndata, scanpy, pandas,
scikit-learn and torch. Competitors additionally need scvi-tools (scANVI),
scarches (scPoli), symphonypy with harmonypy<1 (Symphony), and R with Seurat,
zellkonverter and SingleCellExperiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "datasets"
RESULT_DIR = HERE / "results"
SOURCE_DATA = HERE / "fig2_label_transfer_source_data.csv"
SEURAT_SCRIPT = HERE / "run_seurat.R"
MANIFEST_PATH = HERE / "datasets_manifest.json"

# Frozen inputs live on Figshare. File ids are resolved from the article at run
# time rather than hard-coded, so re-uploading a file does not break the script.
FIGSHARE_ARTICLE = "30671669"
# page_size is required: the API defaults to 10 files per page and would
# silently hide later uploads.
FIGSHARE_API = (
    f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}/files?page_size=1000"
)
FIGSHARE_URL = (
    "https://figshare.com/articles/dataset/Anndata_for_scBIOT_analysis/30671669"
)

# Tabula Sapiens exceeds the Figshare upload limit, so it is rebuilt locally from
# the published atlas instead of being downloaded frozen.
TABULA_SAPIENS_SOURCE = (
    "https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5"
)
TABULA_SAPIENS_FILE = "tabula_sapiens_v2_slim_with_decontXcounts.h5ad"

# Raw inputs, only needed when rebuilding a dataset from scratch.
RAW_ROOT = HERE.parent.parent / "Fig2_label_transfer"

SEED = 0
N_TOP_GENES = 3000
# Batches smaller than this are pooled for the HVG variance fit only. seurat_v3
# fits a LOESS trend per batch and a handful of cells give it nothing to fit.
MIN_CELLS_PER_HVG_BATCH = 50

SPLIT_COL = "bench_split"
LABEL_COL = "bench_label"
EVAL_COL = "bench_eval"
LABEL_KEY = "semi_cell_type"
UNLABELED = "Unknown"
MISSING = "__no_prediction__"

DATASETS = {
    "Pancreas": dict(
        source=RAW_ROOT / "pancreas/alldata_pancreas.h5ad",
        cell_type_key="celltype",
        batch_key="tech",
        targeted_panel=False,
        n_comps=50,
    ),
    "PBMCs": dict(
        source=RAW_ROOT / "PBMC/alldata_pbmc.h5ad",
        cell_type_key="celltype.l2",
        batch_key="orig.ident",
        targeted_panel=False,
        n_comps=50,
    ),
    "Tabula_Muris": dict(
        source=RAW_ROOT / "tabula_muris/alldata_Tabula_Muris.h5ad",
        cell_type_key="cell_ontology_class",
        batch_key="mouse.id",
        targeted_panel=False,
        n_comps=100,
    ),
    "Tabula_Sapiens": dict(
        source=RAW_ROOT / "tabula_sapiens/alldata_Tabula_Sapiens_full.h5ad",
        cell_type_key="true_cell_type",
        batch_key="sample_id",
        targeted_panel=False,
        n_comps=100,
    ),
    "COAD": dict(
        source=RAW_ROOT / "spatial/xenium/COAD_transcriptome/alldata_xenium_COAD.h5ad",
        cell_type_key="cell_type",
        batch_key="modality",
        targeted_panel=True,
        n_comps=100,
    ),
    "HCC": dict(
        source=RAW_ROOT / "spatial/xenium/HCC_transcriptome/alldata_xenium_HCC.h5ad",
        cell_type_key="cell_type",
        batch_key="modality",
        targeted_panel=True,
        n_comps=100,
    ),
    "OV": dict(
        source=RAW_ROOT / "spatial/xenium/OV_transcriptome/alldata_xenium_OV.h5ad",
        cell_type_key="cell_type",
        batch_key="modality",
        targeted_panel=True,
        n_comps=100,
    ),
}

METHODS = ["supBIOT", "Unintegrated", "Symphonypy", "scANVI", "scPoli", "Seurat"]

# Each method runs under an interpreter that can import its dependencies.
# Override these defaults with the corresponding environment variable.
INTERPRETERS = {
    "supBIOT": os.environ.get("SCBIOT_BENCH_PYTHON", sys.executable),
    "Unintegrated": os.environ.get("SCBIOT_BENCH_PYTHON", sys.executable),
    "scANVI": os.environ.get(
        "SCBIOT_SCANVI_PYTHON", str(HERE / "venv_scanvi/bin/python")
    ),
    "scPoli": os.environ.get(
        "SCBIOT_SCPOLI_PYTHON", str(HERE / "venv_scpoli/bin/python")
    ),
    "Symphonypy": os.environ.get(
        "SCBIOT_SYMPHONY_PYTHON", str(HERE / "venv_symphony/bin/python")
    ),
}

RUN_INFO: dict = {}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def figshare_listing() -> dict:
    """Map filename -> (download_url, md5) for the Figshare article."""
    with urllib.request.urlopen(FIGSHARE_API) as response:
        files = json.load(response)
    return {f["name"]: (f["download_url"], f.get("supplied_md5") or f.get("computed_md5"))
            for f in files}


def ensure_dataset(name: str) -> Path:
    """Return the frozen input, downloading it from Figshare when absent."""
    destination = DATA_DIR / f"{name}.h5ad"
    if destination.is_file():
        return destination

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if name == "Tabula_Sapiens":
        raise SystemExit(
            f"{destination} is missing.\n"
            f"Tabula Sapiens is too large to host on Figshare. Download\n"
            f"  {TABULA_SAPIENS_FILE}\n"
            f"from {TABULA_SAPIENS_SOURCE}\n"
            f"into {RAW_ROOT / 'tabula_sapiens'}, then run:\n"
            f"  python {Path(__file__).name} --rebuild Tabula_Sapiens\n"
            f"  python {Path(__file__).name} --dataset Tabula_Sapiens --freeze"
        )

    listing = figshare_listing()
    if destination.name not in listing:
        raise SystemExit(
            f"{destination.name} is not in Figshare article {FIGSHARE_ARTICLE}.\n"
            f"See {FIGSHARE_URL}, or rebuild it locally with --rebuild."
        )

    url, expected = listing[destination.name]
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {destination.name} from {url}", flush=True)
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        observed = file_md5(temporary)
        if expected and observed != expected:
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}: {observed} != {expected}"
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"verified {destination.name}", flush=True)
    return destination


# --------------------------------------------------------------------------- #
# frozen evaluation protocol
# --------------------------------------------------------------------------- #
def sha(values) -> str:
    """Order-sensitive digest of a sequence, for manifest checks."""
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def eval_mask(obs: pd.DataFrame) -> np.ndarray:
    """Cells entering the metric. Depends only on the split and ground truth."""
    return ((obs[SPLIT_COL] == "query") & obs[LABEL_COL].notna()).to_numpy()


def score(obs: pd.DataFrame, pred_col: str, method: str) -> dict:
    from sklearn.metrics import (
        adjusted_rand_score,
        f1_score,
        normalized_mutual_info_score,
    )

    mask = eval_mask(obs)
    y_true = obs.loc[mask, LABEL_COL].astype(str)
    y_pred = obs.loc[mask, pred_col].astype("string").fillna(MISSING).astype(str)
    covered = int((y_pred != MISSING).sum())

    return {
        "method": method,
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "n_cells": int(mask.sum()),
        "n_predicted": covered,
        "coverage": round(covered / max(int(mask.sum()), 1), 6),
    }


# --------------------------------------------------------------------------- #
# dataset construction
# --------------------------------------------------------------------------- #
def rebuild_xenium(cancer: str) -> None:
    """Build alldata for a Xenium panel from its reference and query."""
    import anndata
    import scanpy as sc

    root = RAW_ROOT / "spatial"
    ref = sc.read_h5ad(root / f"reference/{cancer}_transcriptome/adata.h5ad")
    query_dir = root / f"xenium/{cancer}_transcriptome"
    query = sc.read_h5ad(query_dir / "adata.h5ad")

    query.obs["cell_type"] = query.obs["annotation"].copy()
    ref.obs["cell_type"] = ref.obs["major_annotation"]
    query.layers["counts"] = query.X.copy()
    ref.layers["counts"] = ref.X.copy()

    # The reference carries duplicate gene symbols; sum them so the inner join
    # against the panel is well defined.
    ref.var["gene_symbol"] = ref.var_names
    ref = sc.get.aggregate(ref, by="gene_symbol", axis="var", func="sum", layer="counts")
    ref.layers["counts"] = ref.layers["sum"]
    ref.X = ref.layers["sum"]

    ref.obs[LABEL_KEY] = ref.obs["cell_type"]
    query.obs[LABEL_KEY] = UNLABELED
    adata = anndata.concat(
        [ref, query], join="inner", label="modality", keys=["reference", "query"]
    )
    adata.X = adata.layers["counts"].copy()
    out = query_dir / f"alldata_xenium_{cancer}.h5ad"
    adata.write_h5ad(out)
    print(f"wrote {out} {adata.shape}", flush=True)


def rebuild_tabula_muris() -> None:
    """FACS is the reference, droplet the query, with gene-length correction."""
    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    d = RAW_ROOT / "tabula_muris"
    droplet = sc.read_h5ad(d / "tabula-muris-senis-droplet-official-raw-obj.h5ad")
    facs = sc.read_h5ad(d / "tabula-muris-senis-facs-official-raw-obj.h5ad")

    gene_len = pd.read_csv(
        "https://raw.githubusercontent.com/chenlingantelope/HarmonizationSCANVI/"
        "master/data/gene_len.txt",
        sep=" ",
        header=None,
        index_col=0,
    )
    gene_len = gene_len.reindex(facs.var_names).dropna()
    facs = facs[:, gene_len.index].copy()
    factor = np.median(gene_len[1].to_numpy()) / gene_len[1].to_numpy()
    if sparse.issparse(facs.X):
        scaled = facs.X.tocsr().dot(sparse.diags(factor)).tocsr()
        scaled.data = np.rint(scaled.data).astype(np.int64)
        facs.X = scaled
    else:
        facs.X = np.rint(np.asarray(facs.X) * factor).astype(np.int64)

    droplet.layers["counts"] = droplet.X.copy()
    facs.layers["counts"] = facs.X.copy()
    facs.obs[LABEL_KEY] = facs.obs["cell_ontology_class"].astype(str)
    droplet.obs[LABEL_KEY] = UNLABELED

    combined = ad.concat(
        [droplet, facs], label="modality", keys=["query", "reference"], join="inner"
    )
    sc.pp.calculate_qc_metrics(combined, inplace=True)
    sc.pp.filter_cells(combined, min_counts=np.quantile(combined.obs["total_counts"], 0.01))
    sc.pp.filter_cells(combined, max_counts=np.quantile(combined.obs["total_counts"], 0.98))
    sc.pp.filter_genes(combined, min_cells=int(0.01 * combined.n_obs))
    combined.X = combined.layers["counts"].copy()

    out = d / "alldata_Tabula_Muris.h5ad"
    combined.write_h5ad(out)
    print(f"wrote {out} {combined.shape}", flush=True)


def rebuild_tabula_sapiens() -> None:
    """Full atlas, donor TSP14 held out as the query.

    The notebook subsamples to 10% for speed; that is not applied here. `batch`
    is `sample_id`, while `modality` is the reference/query split used for
    evaluation rather than a batch variable to integrate over.
    """
    import scanpy as sc

    d = RAW_ROOT / "tabula_sapiens"
    adata = sc.read_h5ad(d / "tabula_sapiens_v2_slim_with_decontXcounts.h5ad")

    def bool_mask(s):
        if s.dtype == bool:
            return s.fillna(False).to_numpy()
        return (
            s.astype("string").str.lower().isin(["true", "1", "yes"]).fillna(False).to_numpy()
        )

    obs = adata.obs
    mask = obs["donor_id"].notna().to_numpy() & obs["cell_type"].notna().to_numpy()
    if "is_primary_data" in obs:
        mask &= bool_mask(obs["is_primary_data"])
    if "manually_annotated" in obs:
        mask &= bool_mask(obs["manually_annotated"])
    adata = adata[mask].copy()

    adata.obs["true_cell_type"] = adata.obs["cell_type"].astype("string").astype(str)
    is_query = adata.obs["donor_id"].astype("string").eq("TSP14").fillna(False).to_numpy()
    adata.obs["modality"] = np.where(is_query, "query", "reference")
    adata.obs[LABEL_KEY] = adata.obs["true_cell_type"].astype(str)
    adata.obs.loc[is_query, LABEL_KEY] = UNLABELED

    ref_counts = adata.obs.loc[~is_query, "true_cell_type"].value_counts()
    adata = adata[adata.obs["true_cell_type"].isin(ref_counts[ref_counts >= 10].index)].copy()

    adata.layers["counts"] = adata.layers["decontXcounts"].copy()
    adata.X = adata.layers["counts"].copy()

    out = d / "alldata_Tabula_Sapiens_full.h5ad"
    adata.write_h5ad(out)
    print(f"wrote {out} {adata.shape}", flush=True)


def freeze(name: str) -> dict:
    """Preprocess once and write the frozen input plus its manifest."""
    import scanpy as sc

    cfg = DATASETS[name]
    hvg_note = "n/a (targeted panel)"
    adata = sc.read_h5ad(cfg["source"])

    # Artifacts of earlier runs must not leak in as inputs. An n_obs x n_obs
    # neighbour graph in obsp also makes any later subset densify to n_query^2.
    if adata.obsp:
        del adata.obsp
    for stale in ("X_ae", "supBIOT", "supBIOT_one_stage", "X_umap_supBIOT"):
        adata.obsm.pop(stale, None)

    if cfg["targeted_panel"]:
        # Spatial QC on the query arm, computed on the sparse counts directly.
        is_query = (adata.obs["modality"] == "query").to_numpy()
        q_counts = adata.layers["counts"][is_query]
        total = np.asarray(q_counts.sum(axis=1)).ravel()
        keep_q = total >= 20
        keep_q &= total <= np.quantile(total[keep_q], 0.98)
        per_gene = np.asarray((q_counts[keep_q] > 0).sum(axis=0)).ravel()
        keep_cells = ~is_query
        keep_cells[is_query] = keep_q
        adata = adata[keep_cells, per_gene >= 100].copy()
    elif adata.n_vars > N_TOP_GENES:
        batch = adata.obs[cfg["batch_key"]].astype(str)
        counts = batch.value_counts()
        small = counts[counts < MIN_CELLS_PER_HVG_BATCH].index
        if len(small):
            batch = batch.mask(batch.isin(small), "__pooled_small_batches__")
            hvg_note = f"pooled {len(small)} batches with <{MIN_CELLS_PER_HVG_BATCH} cells"
        else:
            hvg_note = "per-batch"
        adata.obs["_hvg_batch"] = batch.astype("category")
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=N_TOP_GENES,
                flavor="seurat_v3",
                layer="counts",
                span=0.6,
                batch_key="_hvg_batch",
            )
        except Exception as exc:  # noqa: BLE001 - loess raises a bare ValueError
            print(f"[{name}] batched HVG failed ({exc}); retrying unbatched", flush=True)
            sc.pp.highly_variable_genes(
                adata, n_top_genes=N_TOP_GENES, flavor="seurat_v3", layer="counts", span=0.6
            )
            hvg_note = f"unbatched (batched fit failed: {exc})"
        del adata.obs["_hvg_batch"]
        adata = adata[:, adata.var["highly_variable"]].copy()

    adata.obs[SPLIT_COL] = adata.obs["modality"].astype(str)
    # Categorical, not nullable string: anndata cannot write a StringArray that
    # carries missing values, and unlabelled query cells are exactly that.
    adata.obs[LABEL_COL] = adata.obs[cfg["cell_type_key"]].astype("object").astype("category")
    adata.obs[EVAL_COL] = eval_mask(adata.obs)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(DATA_DIR / f"{name}.h5ad")

    ev = adata.obs[EVAL_COL].to_numpy()
    manifest = {
        "dataset": name,
        "seed": SEED,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_reference": int((adata.obs[SPLIT_COL] == "reference").sum()),
        "n_query": int((adata.obs[SPLIT_COL] == "query").sum()),
        "n_eval": int(ev.sum()),
        "n_classes": int(adata.obs.loc[ev, LABEL_COL].nunique()),
        "targeted_panel": cfg["targeted_panel"],
        "batch_key": cfg["batch_key"],
        "label_source": cfg["cell_type_key"],
        "hvg_batching": hvg_note,
        "sha_obs_names": sha(adata.obs_names),
        "sha_var_names": sha(adata.var_names),
        "sha_eval_cells": sha(adata.obs_names[ev]),
        "sha_labels": sha(adata.obs.loc[ev, LABEL_COL]),
    }
    all_manifests = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    all_manifests[name] = manifest
    MANIFEST_PATH.write_text(json.dumps(all_manifests, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
def run_supbiot(adata, cfg, smoke=False):
    """Supervised autoencoder, OT integration at defaults, supBIOT readout."""
    import scbiot as scb

    # LBFGS is full-batch and deterministic and is used wherever it fits. On the
    # largest atlases the gradient exceeds GPU memory; fall back to minibatch
    # Adam in that case only. The criterion is memory, never the score, and the
    # solver actually used is recorded in the result.
    try:
        adata = scb.pp.autoencoder(
            adata,
            input_key="counts",
            out_key="X_ae",
            label_key=LABEL_KEY,
            unlabeled_category=UNLABELED,
            solver="lbfgs",
        )
        RUN_INFO["solver"] = "lbfgs"
    except Exception as exc:  # noqa: BLE001 - torch raises several OOM types
        if "out of memory" not in str(exc).lower():
            raise
        import torch

        torch.cuda.empty_cache()
        print(f"[supBIOT] LBFGS OOM, falling back to Adam: {exc}", flush=True)
        adata = scb.pp.autoencoder(
            adata,
            input_key="counts",
            out_key="X_ae",
            label_key=LABEL_KEY,
            unlabeled_category=UNLABELED,
            solver="adam",
        )
        RUN_INFO["solver"] = "adam (lbfgs OOM)"

    adata, _ = scb.ot.integrate(
        adata,
        obsm_key="X_ae",
        batch_key=cfg["batch_key"],
        label_key=LABEL_KEY,
        unlabeled_category=UNLABELED,
        out_key="supBIOT",
    )
    adata = scb.ot.supbiot(
        adata,
        input_rep_key="X_ae",
        use_rep="supBIOT",
        label_key=LABEL_KEY,
        unlabeled_category=UNLABELED,
        pred_label_key="pred_cell_type",
        pred_conf_key="pred_confidence",
        min_conf=0.0,
    )
    return adata.obs["pred_cell_type"]


def run_unintegrated(adata, cfg, smoke=False):
    """Multinomial logistic classifier on unintegrated PCA.

    This is the supBIOT readout with its auxiliary representation and prototype
    blends disabled, so it stays a strict PCA baseline rather than a hybrid.
    """
    import scanpy as sc
    from scbiot.ot.supbiot import predict_pseudo_labels

    n_comps = 20 if smoke else cfg["n_comps"]
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_comps, random_state=SEED)
    adata.obsm["Unintegrated"] = np.asarray(adata.obsm["X_pca"], dtype=np.float32)

    pred, _ = predict_pseudo_labels(
        adata,
        rep_key="Unintegrated",
        label_key=LABEL_KEY,
        unlabeled_category=UNLABELED,
        min_conf=0.0,
        return_numpy=True,
        inplace=False,
        max_ref=20_000,
        use_gpu=True,
        gpu_device=0,
        transfer_mode="logreg",
        logreg_tissue_key="tissue",
        logreg_min_cells=80,
        logreg_min_classes=2,
        logreg_class_weight="balanced",
        logreg_C=5.0,
        logreg_solver="lbfgs",
        logreg_penalty="l2",
        logreg_multi_class="multinomial",
        logreg_max_iter=3000,
        logreg_prior_alpha=0.0,
        logreg_standardize=True,
        input_rep_key=None,
        input_rep_weight=0.0,
        prototype_weight=0.0,
        query_group_key="modality",
        query_group_value="query",
        random_state=SEED,
    )
    return pd.Series(np.asarray(pred, dtype=object), index=adata.obs_names)


def run_symphony(adata, cfg, smoke=False):
    """Harmony on the reference, then Symphony mapping and kNN label transfer.

    The original protocol re-selects HVGs here. The frozen file has already
    fixed the feature set, so every frozen gene is marked variable instead,
    keeping the pipeline shape without letting features drift between methods.
    """
    import harmonypy as hm
    import scanpy as sc
    import symphonypy as sp

    np.random.seed(SEED)
    n_comps = 20 if smoke else cfg["n_comps"]
    adata.X = adata.layers["counts"]
    ref = adata[adata.obs[SPLIT_COL] == "reference"].copy()
    query = adata[adata.obs[SPLIT_COL] == "query"].copy()

    sc.pp.normalize_total(ref)
    sc.pp.log1p(ref)
    ref.var["highly_variable"] = True
    sc.pp.scale(ref)
    sc.pp.pca(ref, n_comps=n_comps, use_highly_variable=True)

    ho = hm.run_harmony(ref.obsm["X_pca"], meta_data=ref.obs, vars_use=cfg["batch_key"])
    if ho.Z_corr.shape[0] != ref.n_obs:
        Z_corr, R = np.asarray(ho.Z_corr.T), np.asarray(ho.R)
    else:
        Z_corr, R = np.asarray(ho.Z_corr), np.asarray(ho.R.T)
    ref.obsm["X_pca_harmony"] = Z_corr
    ref.uns["harmony"] = {
        "Nr": R.sum(axis=1),
        "C": R @ Z_corr,
        "K": ho.K,
        "sigma": ho.sigma,
        "ref_basis_loadings": "PCs",
        "ref_basis_adjusted": "X_pca_harmony",
        "vars_use": cfg["batch_key"],
        "harmony_kwargs": {},
        "converged": ho.check_convergence(1),
        "R": R,
    }

    sc.pp.normalize_total(query)
    sc.pp.log1p(query)
    sp.tl.map_embedding(adata_query=query, adata_ref=ref, key=cfg["batch_key"])
    sc.pp.neighbors(ref, use_rep="X_pca_harmony")
    sc.tl.umap(ref)
    sp.tl.ingest(adata_query=query, adata_ref=ref, use_rep="X_pca_harmony")
    sp.tl.transfer_labels_kNN(
        adata_query=query,
        adata_ref=ref,
        ref_labels=[LABEL_KEY],
        query_labels=["C_Symphonypy"],
        ref_basis="X_pca_harmony",
        query_basis="X_pca_harmony",
    )
    return pd.Series(query.obs["C_Symphonypy"].astype(str).to_numpy(), index=query.obs_names)


def run_scanvi(adata, cfg, smoke=False):
    """SCVI then SCANVI with a linear classifier, as in the original notebooks."""
    import scvi

    scvi.settings.seed = SEED
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=cfg["batch_key"])
    vae = scvi.model.SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=30)
    vae.train(batch_size=512, max_epochs=2 if smoke else None)

    lvae = scvi.model.SCANVI.from_scvi_model(
        vae,
        adata=adata,
        labels_key=LABEL_KEY,
        unlabeled_category=UNLABELED,
        linear_classifier=True,
    )
    lvae.train(max_epochs=2 if smoke else 20, n_samples_per_label=100, batch_size=128)
    return pd.Series(lvae.predict(adata), index=adata.obs_names)


def run_scpoli(adata, cfg, smoke=False):
    import scipy.sparse as sparse
    import torch
    from scarches.models.scpoli import scPoli

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    adata.X = adata.layers["counts"]
    adata.X = (
        adata.X.astype(np.float32)
        if sparse.issparse(adata.X)
        else np.asarray(adata.X, dtype=np.float32)
    )

    model = scPoli(
        adata=adata,
        condition_keys=cfg["batch_key"],
        cell_type_keys=LABEL_KEY,
        unknown_ct_names=[UNLABELED],
        embedding_dims=5,
        recon_loss="nb",
    )
    n_epochs, pre = (3, 2) if smoke else (50, 40)
    model.train(n_epochs=n_epochs, pretraining_epochs=pre, eta=5)
    res = model.classify(adata)
    return pd.Series(res[LABEL_KEY]["preds"].astype(str), index=adata.obs_names)




def ensure_cells_csv(dataset: str) -> Path:
    """Derive the cell order Seurat needs, since zellkonverter drops the index.

    Regenerated from the frozen h5ad rather than shipped, so `datasets/` holds
    only the h5ad files themselves.
    """
    import anndata as ad

    path = RESULT_DIR / f"{dataset}.cells.csv"
    if not path.exists():
        obs = ad.read_h5ad(ensure_dataset(dataset), backed="r").obs
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        pd.Series(obs.index, name="cell").to_csv(path, index=False)
    return path


def load_manifest(dataset: str) -> dict:
    """Manifest for one dataset, rebuilding the digests if it is absent."""
    if MANIFEST_PATH.exists():
        manifests = json.loads(MANIFEST_PATH.read_text())
        if dataset in manifests:
            return manifests[dataset]
    return freeze(dataset)


def run_seurat_via_r(dataset: str) -> pd.Series:
    """Run run_seurat.R and read back its per-cell predictions."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_cells_csv(dataset)
    subprocess.run(
        ["Rscript", str(SEURAT_SCRIPT), dataset, str(DATA_DIR), str(RESULT_DIR)],
        check=True,
    )
    pred = pd.read_csv(RESULT_DIR / f"{dataset}_Seurat.pred.csv", index_col=0)
    return pred["prediction"]


RUNNERS = {
    "supBIOT": run_supbiot,
    "Unintegrated": run_unintegrated,
    "Symphonypy": run_symphony,
    "scANVI": run_scanvi,
    "scPoli": run_scpoli,
}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_one(dataset: str, method: str, smoke: bool = False) -> dict:
    """Execute one method in the current interpreter and write its result."""
    import anndata as ad
    import scanpy as sc

    cfg = DATASETS[dataset]
    manifest = load_manifest(dataset)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if method == "Seurat":
        if smoke:
            # run_seurat.R has no subsampling path, so a "smoke" Seurat run would
            # quietly be a full one. Say so rather than burn the time.
            print("smoke mode is not supported for Seurat; skipping", flush=True)
            return {}
        pred = run_seurat_via_r(dataset)
        obs = ad.read_h5ad(ensure_dataset(dataset), backed="r").obs
    else:
        adata = sc.read_h5ad(ensure_dataset(dataset))
        if smoke:
            sc.pp.subsample(adata, n_obs=min(3000, adata.n_obs), random_state=SEED)
        else:
            # The frozen file is the contract. Verify before spending compute.
            assert sha(adata.obs_names) == manifest["sha_obs_names"], "cells drifted"
            assert sha(adata.var_names) == manifest["sha_var_names"], "features drifted"
        pred = RUNNERS[method](adata, cfg, smoke)
        obs = adata.obs

    obs[f"C_{method}"] = pred.reindex(obs.index)
    res = score(obs, f"C_{method}", method)
    res.update(
        dataset=dataset,
        runtime_s=round(time.time() - t0, 1),
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
    )
    try:
        import torch

        if torch.cuda.is_available():
            res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except ImportError:
        pass
    res.update(RUN_INFO)

    # Seurat measures its own time and memory on the R side.
    resources = RESULT_DIR / f"{dataset}_{method}.resources.csv"
    if resources.exists():
        res.update(pd.read_csv(resources).iloc[0].to_dict())

    tag = f"smoke_{dataset}" if smoke else dataset
    pred.rename("prediction").to_frame().to_csv(RESULT_DIR / f"{tag}_{method}.pred.csv")
    (RESULT_DIR / f"{tag}_{method}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


def dispatch(dataset: str, method: str, smoke: bool) -> bool:
    """Re-invoke this file under the interpreter the method needs."""
    out = RESULT_DIR / f"{dataset}_{method}.json"
    if out.exists() and not smoke:
        print(f"skip {dataset}/{method} (done)", flush=True)
        return True

    interpreter = INTERPRETERS.get(method, sys.executable)
    if method != "Seurat" and not Path(interpreter).exists():
        print(f"skip {dataset}/{method}: interpreter not found ({interpreter})", flush=True)
        return False

    cmd = [interpreter, __file__, "--dataset", dataset, "--method", method, "--worker"]
    if smoke:
        cmd.append("--smoke")
    print(f"run  {dataset}/{method}", flush=True)
    return subprocess.run(cmd).returncode == 0


def write_source_data() -> None:
    rows = []
    for f in sorted(RESULT_DIR.glob("*.json")):
        if f.name.startswith("smoke_"):
            continue
        rows.append(json.load(open(f)))
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = ["dataset", "method", "F1_macro", "F1_weighted", "NMI", "ARI",
            "n_cells", "n_predicted", "coverage", "runtime_s"]
    cols += [c for c in df.columns if c not in cols]
    df[cols].sort_values(["dataset", "method"]).to_csv(SOURCE_DATA, index=False)
    print(f"wrote {SOURCE_DATA} ({len(df)} runs)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--all", action="store_true", help="run every dataset and method")
    parser.add_argument("--rebuild", choices=["HCC", "Tabula_Muris", "Tabula_Sapiens"],
                        help="rebuild a dataset from raw inputs before freezing")
    parser.add_argument("--freeze", action="store_true", help="(re)freeze before running")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny subsample and few epochs, to validate an environment")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.rebuild:
        {"HCC": lambda: rebuild_xenium("HCC"),
         "Tabula_Muris": rebuild_tabula_muris,
         "Tabula_Sapiens": rebuild_tabula_sapiens}[args.rebuild]()
        return

    if args.worker:
        run_one(args.dataset, args.method, args.smoke)
        return

    datasets = [args.dataset] if args.dataset else list(DATASETS)
    methods = [args.method] if args.method else METHODS

    for dataset in datasets:
        ensure_dataset(dataset)
        manifests = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
        if args.freeze or dataset not in manifests:
            print(f"freezing {dataset}", flush=True)
            freeze(dataset)
        # Serial by design: two large jobs at once exhaust memory on one host.
        for method in methods:
            dispatch(dataset, method, args.smoke)

    write_source_data()


if __name__ == "__main__":
    main()
