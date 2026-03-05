#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import scanpy as sc

from scbiot.ot import integrate as integrate_api
import scbiot.ot.integrate as integrate_mod


def _run_once(adata, out_key: str, **kwargs):
    adata_run = adata.copy()
    _, metrics = integrate_api(adata_run, out_key=out_key, **kwargs)
    return metrics


def _format_metrics(metrics: dict) -> str:
    parts = []
    for key in sorted(metrics):
        val = metrics[key]
        if isinstance(val, float):
            parts.append(f"{key}={val:.4f}")
        else:
            parts.append(f"{key}={val}")
    return " ".join(parts)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare LSI cluster-debatch against baseline.")
    parser.add_argument("--input", required=True, help="Path to input .h5ad")
    parser.add_argument("--preset", default="atac", help="Integration preset (default: atac)")
    parser.add_argument("--obsm-key", default=None, help="Override embedding key")
    parser.add_argument("--batch-key", default=None, help="Override batch key")
    parser.add_argument("--out-key-base", default="X_ot_base", help="Baseline output key")
    parser.add_argument("--out-key-cluster", default="X_ot_cluster", help="Cluster-debatch output key")
    parser.add_argument("--max-iter", type=int, default=None, help="Override max_iter")
    parser.add_argument("--eval-subsample", type=int, default=None, help="Override eval_subsample")
    parser.add_argument("--trust-subsample", type=int, default=None, help="Override trust_subsample")
    parser.add_argument("--approximate-ot", action="store_true", help="Use approximate OT")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    args = parser.parse_args(argv)

    adata = sc.read_h5ad(args.input)

    run_kwargs = {
        "preset": args.preset,
        "projector_strength": 0.0,
        "approximate_ot": args.approximate_ot,
    }
    if args.obsm_key:
        run_kwargs["obsm_key"] = args.obsm_key
    if args.batch_key:
        run_kwargs["batch_key"] = args.batch_key
    if args.max_iter is not None:
        run_kwargs["max_iter"] = args.max_iter
    if args.eval_subsample is not None:
        run_kwargs["eval_subsample"] = args.eval_subsample
    if args.trust_subsample is not None:
        run_kwargs["trust_subsample"] = args.trust_subsample
    if args.no_gpu:
        run_kwargs["use_gpu"] = False

    debatch_max = integrate_mod._DEBATCH_MAX
    try:
        integrate_mod._DEBATCH_MAX = 0.0
        base_metrics = _run_once(adata, out_key=args.out_key_base, **run_kwargs)
    finally:
        integrate_mod._DEBATCH_MAX = debatch_max

    cluster_metrics = _run_once(adata, out_key=args.out_key_cluster, **run_kwargs)

    print("[baseline]", _format_metrics(base_metrics))
    print("[cluster]", _format_metrics(cluster_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
