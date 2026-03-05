#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc

from scbiot.ot._presets import get_modality_preset
import scbiot.ot.integrate as integrate_mod


def _run_integration(adata, params, preset):
    adata_run = adata.copy()
    _, metrics = integrate_mod.integrate_ot(adata_run, preset=preset, **params)
    return adata_run, metrics


def _save_umap(adata, out_key: str, batch_key: str, fig_path: Path, n_neighbors: int, min_dist: float):
    sc.pp.neighbors(adata, use_rep=out_key, n_neighbors=n_neighbors, metric="cosine")
    sc.tl.umap(adata, min_dist=min_dist)
    color_key = batch_key if batch_key in adata.obs else None
    sc.pl.umap(adata, color=color_key, show=False)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def _format_metrics(metrics: dict) -> str:
    parts = []
    for key in ("mix", "overlap0", "strain", "tw", "it"):
        val = metrics.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            parts.append(f"{key}={val:.4f}")
        else:
            parts.append(f"{key}={val}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="ATAC LSI stability comparison with/without LSI-safe patches.")
    parser.add_argument("--input", required=True, help="Path to input .h5ad")
    parser.add_argument("--obsm-key", default="X_lsi", help="Embedding key for ATAC LSI")
    parser.add_argument("--batch-key", default="batchname_all", help="Batch key")
    parser.add_argument("--out-dir", default="umap_out", help="Directory for UMAP figures")
    parser.add_argument("--n-neighbors", type=int, default=50, help="Neighbors for UMAP")
    parser.add_argument("--min-dist", type=float, default=0.3, help="UMAP min_dist")
    parser.add_argument("--max-iter", type=int, default=None, help="Override max_iter")
    parser.add_argument("--eval-subsample", type=int, default=None, help="Override eval_subsample")
    parser.add_argument("--trust-subsample", type=int, default=None, help="Override trust_subsample")
    parser.add_argument("--approximate-ot", action="store_true", help="Use approximate OT")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.input)

    base_params = dict(get_modality_preset("atac"))
    base_params["batch_key"] = args.batch_key
    base_params["obsm_key"] = args.obsm_key
    base_params["approximate_ot"] = args.approximate_ot
    if args.max_iter is not None:
        base_params["max_iter"] = args.max_iter
    if args.eval_subsample is not None:
        base_params["eval_subsample"] = args.eval_subsample
    if args.trust_subsample is not None:
        base_params["trust_subsample"] = args.trust_subsample
    if args.no_gpu:
        base_params["use_gpu"] = False

    alias_key = "X_embed"
    adata_base = adata.copy()
    adata_base.obsm[alias_key] = adata_base.obsm[args.obsm_key].copy()

    base_run_params = dict(base_params)
    base_run_params["obsm_key"] = alias_key
    base_run_params["out_key"] = "X_ot_base"
    adata_base_run, base_metrics = _run_integration(adata_base, base_run_params, preset=None)

    patch_params = dict(base_params)
    patch_params["out_key"] = "X_ot_patch"
    adata_patch_run, patch_metrics = _run_integration(adata, patch_params, preset="atac")

    out_dir = Path(args.out_dir)
    _save_umap(
        adata_base_run,
        out_key="X_ot_base",
        batch_key=args.batch_key,
        fig_path=out_dir / "umap_baseline.png",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    _save_umap(
        adata_patch_run,
        out_key="X_ot_patch",
        batch_key=args.batch_key,
        fig_path=out_dir / "umap_patch.png",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    print("[baseline]", _format_metrics(base_metrics))
    print("[patch]", _format_metrics(patch_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
