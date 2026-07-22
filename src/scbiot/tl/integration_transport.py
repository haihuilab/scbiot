"""
Explainable batch integration — gene-level attribution of an scBIOT integration.

scBIOT aligns batches with an *explicit* optimal-transport coupling, so unlike
black-box integrators the correction it applies can be audited per gene. This
module turns that coupling into an **integration transport score**: how much each
gene's expression must shift to map every batch onto a shared reference in the
integrated embedding.

    high transport  -> gene the integration MOVED   (batch / technical variation)
    low  transport  -> gene the integration KEPT    (conserved biology)

The coupling is built in the *integrated* representation (so it matches cells by
biology across batches) with scBIOT's own entropic-OT solver
(:func:`scbiot.utils.ot_transport.compute_ot_alignment`); log-normalized
expression is then barycentrically projected through it. Expression is z-scored
per gene by default so the score measures *relative* movement — without this,
ultra-high-expression genes dominate the absolute shift and get mislabelled as
technical even when they are cell-type markers.

This is distinct from :func:`scbiot.tl.rank_transport_score`, whose signed,
variance-normalized aggregation is designed for ordered *trajectory* states and
does not recover batch genes from a batch coupling.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from ..utils.ot_transport import compute_ot_alignment

__all__ = ["integration_transport_score", "per_batch_transport"]


def _dense(X) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _expression(adata, layer: Optional[str], standardize: bool) -> np.ndarray:
    X = adata.layers[layer] if layer is not None else adata.X
    L = _dense(X).astype(np.float64)
    if standardize:
        L = (L - L.mean(axis=0)) / (L.std(axis=0) + 1e-8)
    return L


def _coupling_matrix(source_rep, ref_rep, *, use_ot=True, k=15, **ot_kwargs):
    """Row-stochastic barycentric map (n_source x n_ref) from source onto ref.

    Uses scBIOT's entropic-OT plan when ``use_ot`` (genuine transport); falls back
    to a symmetric kNN barycenter otherwise. Rows sum to 1.
    """
    n_s, n_r = source_rep.shape[0], ref_rep.shape[0]
    if use_ot:
        try:
            _, transport = compute_ot_alignment(
                source_rep.astype(np.float32), ref_rep.astype(np.float32),
                use_gpu=True, **ot_kwargs,
            )
            idx = np.asarray(transport["indices"], dtype=np.int64)
            w = np.asarray(transport["weights"], dtype=np.float64)
            rows = np.repeat(np.arange(n_s), idx.shape[1])
            M = sparse.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n_s, n_r))
            rs = np.asarray(M.sum(1)).ravel()
            rs[rs == 0] = 1.0
            return M.multiply(1.0 / rs[:, None]).tocsr()
        except Exception as exc:  # pragma: no cover - defensive fallback
            import warnings
            warnings.warn(f"OT coupling failed ({type(exc).__name__}); "
                          f"falling back to kNN barycenter.")
    from sklearn.neighbors import NearestNeighbors

    kk = min(k, n_r)
    nn = NearestNeighbors(n_neighbors=kk).fit(ref_rep)
    _, idx = nn.kneighbors(source_rep)
    rows = np.repeat(np.arange(n_s), kk)
    return sparse.csr_matrix(
        (np.full(rows.size, 1.0 / kk), (rows, idx.ravel())), shape=(n_s, n_r)
    )


def integration_transport_score(
    adata,
    *,
    batch_key: str = "batch",
    rep_key: str = "scBIOT",
    layer: Optional[str] = "lognorm",
    reference: Optional[str] = "auto",
    within: Optional[str] = None,
    standardize: bool = True,
    use_ot: bool = True,
    k: int = 15,
    store_key: str = "integration_transport",
) -> pd.DataFrame:
    """Per-gene transport magnitude under an scBIOT batch integration.

    For every non-reference batch, an OT coupling to the reference is built in
    ``rep_key`` space; log-normalized expression is barycentrically projected
    through it and the mean absolute per-gene shift
    ``mean_i |x_g(i) - x_hat_g(i)|`` is accumulated. Batch/technical genes shift a
    lot (matched cells differ across batches); conserved-biology genes barely move.

    Parameters
    ----------
    adata
        AnnData with the integrated embedding in ``obsm[rep_key]`` and expression
        in ``layers[layer]`` (or ``.X`` if ``layer`` is None), plus ``batch_key``
        in ``.obs``.
    batch_key
        ``.obs`` column defining the batches whose coupling is scored.
    rep_key
        ``.obsm`` key of the embedding used to build the coupling. Use the
        *integrated* embedding (e.g. ``'scBIOT'``) so cells match by biology.
    layer
        Expression layer for the per-gene shift (log-normalized). ``None`` uses ``.X``.
    reference
        Batch value to map every other batch onto, ``'auto'`` (largest batch), or
        ``None`` for all ordered pairs (symmetric).
    within
        Optional ``.obs`` key; only couple cells sharing this label (e.g. keep the
        coupling within a condition so a nuisance batch axis is not scored across a
        biological one). Genes are still pooled across strata.
    standardize
        z-score expression per gene before the shift (recommended; see module
        docstring).
    use_ot
        Use scBIOT's entropic-OT coupling (True) or a kNN barycenter (False).
    k
        Neighbours for the kNN-barycenter fallback / matching.
    store_key
        ``.uns`` key to cache the result under.

    Returns
    -------
    pandas.DataFrame
        Columns ``gene``, ``transport`` (magnitude), ``rank_pct`` (0-1), sorted by
        ``transport`` descending. Also stored in ``adata.uns[store_key]``.
    """
    if rep_key not in adata.obsm:
        raise KeyError(f"rep_key '{rep_key}' not in adata.obsm; run scbiot.ot.integrate first.")
    if batch_key not in adata.obs:
        raise KeyError(f"batch_key '{batch_key}' not in adata.obs.")

    rep = np.asarray(adata.obsm[rep_key], dtype=np.float32)
    L = _expression(adata, layer, standardize)
    genes = np.asarray(adata.var_names)
    bat = adata.obs[batch_key].astype(str).values
    batches = pd.unique(bat)

    strata = (adata.obs[within].astype(str).values if within is not None
              else np.zeros(adata.n_obs, dtype=int))

    if reference == "auto":
        vals, counts = np.unique(bat, return_counts=True)
        reference = str(vals[int(np.argmax(counts))])

    if reference is None:
        pairs = [(a, b) for a in batches for b in batches if a != b]
    else:
        pairs = [(b, reference) for b in batches if b != reference]

    score = np.zeros(adata.n_vars, dtype=np.float64)
    n_used = 0
    for src, ref in pairs:
        for s in np.unique(strata):
            ia = np.where((bat == src) & (strata == s))[0]
            ib = np.where((bat == ref) & (strata == s))[0]
            if len(ia) < k or len(ib) < k:
                continue
            M = _coupling_matrix(rep[ia], rep[ib], use_ot=use_ot, k=k)
            xhat = M @ L[ib]
            score += np.abs(L[ia] - xhat).mean(axis=0)
            n_used += 1
    if n_used == 0:
        raise ValueError("No batch pair had >= k cells; lower k or check `within`.")
    score /= n_used

    df = pd.DataFrame({"gene": genes, "transport": score})
    df["rank_pct"] = df["transport"].rank(pct=True)
    df = df.sort_values("transport", ascending=False).reset_index(drop=True)
    adata.uns[store_key] = {
        "df": df, "rep_key": rep_key, "batch_key": batch_key,
        "reference": reference, "within": within, "use_ot": use_ot,
    }
    return df


def per_batch_transport(
    adata,
    *,
    batch_key: str = "batch",
    rep_key: str = "scBIOT",
    layer: Optional[str] = "lognorm",
    reference: Optional[str] = "auto",
    genes: Optional[list] = None,
    standardize: bool = True,
    use_ot: bool = True,
    k: int = 15,
):
    """Signed mean shift ``x - x_hat`` per (gene, batch) toward the reference.

    Positive = the gene reads higher in ``batch`` than in the reference (the
    direction the integration pushes it). Useful for a "which batch drives each
    gene" heatmap.

    Returns
    -------
    (gene_names, batch_names, matrix)
        ``matrix`` has shape ``(len(gene_names), n_non_reference_batches)``.
    """
    rep = np.asarray(adata.obsm[rep_key], dtype=np.float32)
    L = _expression(adata, layer, standardize)
    all_genes = np.asarray(adata.var_names)
    gidx = (np.arange(adata.n_vars) if genes is None
            else np.array([int(np.where(all_genes == g)[0][0]) for g in genes]))
    bat = adata.obs[batch_key].astype(str).values
    batches = list(pd.unique(bat))
    if reference == "auto":
        vals, counts = np.unique(bat, return_counts=True)
        reference = str(vals[int(np.argmax(counts))])
    others = [b for b in batches if b != reference]

    mat = np.zeros((len(gidx), len(others)))
    ib = np.where(bat == reference)[0]
    for j, b in enumerate(others):
        ia = np.where(bat == b)[0]
        M = _coupling_matrix(rep[ia], rep[ib], use_ot=use_ot, k=k)
        xhat = M @ L[ib]
        mat[:, j] = (L[ia] - xhat).mean(axis=0)[gidx]
    return all_genes[gidx], others, mat
