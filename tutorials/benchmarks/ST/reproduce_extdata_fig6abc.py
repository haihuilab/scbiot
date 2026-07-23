#!/usr/bin/env python3
"""Reproduce Extended Data Fig. 6a panels a-c from the six ST datasets.

The script runs scBIOT-ST, scBIOT without spatial geometry, moscot-spatial,
moscot, Waddington-OT, and Sinkhorn-OT on identical adjacent-stage subsamples.
It writes:

  ExtData_Fig6a_abc.pdf
  ExtData_Fig6a_abc.png
  ExtData_Fig6a_abc_source_data.csv

Run from the repository root:

  python tutorials/benchmarks/ST/reproduce_extdata_fig6abc.py

Datasets are downloaded from Figshare article 30671669 when absent:
https://figshare.com/articles/dataset/Anndata_for_scBIOT_analysis/30671669

Required packages: the local scbiot package plus anndata, scanpy, matplotlib,
moscot, wot, POT, scipy, scikit-learn, torch, and pandas. A CUDA device is used
when available; pass --device cpu to force CPU execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


HERE = Path(__file__).resolve().parent
DATA = HERE / "datasets"
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 0
N_PER_STAGE = 2500
SPATIAL_W = 2.0
EPSILON = 5e-2
FIGSHARE_ARTICLE_URL = "https://figshare.com/articles/dataset/Anndata_for_scBIOT_analysis/30671669"
FIGSHARE_FILES = {
    "MOSTA_checkpoint.npz": {
        "url": "https://ndownloader.figshare.com/files/66868592",
        "md5": "a31a3a2c53677746826901fffedaeaeb",
    },
    "axolotl.h5ad": {
        "url": "https://ndownloader.figshare.com/files/66868598",
        "md5": "610f6794551d0a482c0928f5aa640033",
    },
    "c_elegans.h5ad": {
        "url": "https://ndownloader.figshare.com/files/66868589",
        "md5": "4aad50551615121766f225cac3bba511",
    },
    "sim_branching.h5ad": {
        "url": "https://ndownloader.figshare.com/files/66868583",
        "md5": "cf3e81584f7c1c4d9a954a107c9530e1",
    },
    "sim_spatial.h5ad": {
        "url": "https://ndownloader.figshare.com/files/66868580",
        "md5": "3fd9a8e1d620465d08303009b9e18507",
    },
    "zebrafish.h5ad": {
        "url": "https://ndownloader.figshare.com/files/66868595",
        "md5": "1baf7e40dd2050a112aace56d8cd2fcf",
    },
}

COMPUTE_ORDER = [
    "MOSTA",
    "sim_spatial",
    "sim_branching",
    "c_elegans",
    "zebrafish",
    "axolotl",
]
DISPLAY_NAMES = {
    "MOSTA": "MOSTA",
    "axolotl": "Axolotl",
    "c_elegans": "C. elegans",
    "zebrafish": "Zebrafish",
    "sim_spatial": "Sim-spatial",
    "sim_branching": "Sim-branch",
}
DISPLAY_ORDER = list(DISPLAY_NAMES)
METHODS = [
    "scBIOT-ST",
    "scBIOT-noSpatial",
    "moscot-spatial",
    "moscot",
    "WaddingtonOT",
    "Sinkhorn-OT",
]
SCIMORPH_CYCLE = [
    "#386cb0", "#660066", "#336600", "#D2691E", "#A52A2A", "#FFB90F",
    "#7fc97f", "#fb9a99", "#B8860B", "#CD1076", "#458B74", "#E5CCFF",
]
COLORS = {
    "scBIOT-ST": "#A52A2A",
    "scBIOT-noSpatial": SCIMORPH_CYCLE[0],
    "moscot-spatial": SCIMORPH_CYCLE[1],
    "moscot": SCIMORPH_CYCLE[2],
    "WaddingtonOT": SCIMORPH_CYCLE[3],
    "Sinkhorn-OT": SCIMORPH_CYCLE[5],
}

DATASETS = {
    "MOSTA": {
        "type": "checkpoint",
        "path": DATA / "MOSTA_checkpoint.npz",
        "label": "annotation",
        "pairs": [("E10.5", "E11.5"), ("E12.5", "E13.5"), ("E14.5", "E15.5")],
    },
    "sim_spatial": {"type": "h5ad", "path": DATA / "sim_spatial.h5ad", "label": "lineage"},
    "sim_branching": {"type": "h5ad", "path": DATA / "sim_branching.h5ad", "label": "lineage"},
    "c_elegans": {"type": "h5ad", "path": DATA / "c_elegans.h5ad", "label": "label"},
    "zebrafish": {"type": "h5ad", "path": DATA / "zebrafish.h5ad", "label": "label"},
    "axolotl": {"type": "h5ad", "path": DATA / "axolotl.h5ad", "label": "label"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--n-per-stage", type=int, default=N_PER_STAGE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=HERE / "ExtData_Fig6a_abc")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and checksum-verify the six Figshare datasets, then exit.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Plot a previously written result JSON instead of rerunning the benchmark.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def use_gpu(device: str) -> bool:
    import torch

    available = torch.cuda.is_available()
    if device == "gpu" and not available:
        raise RuntimeError("--device gpu was requested, but CUDA is unavailable")
    return available if device == "auto" else device == "gpu"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_datasets() -> None:
    """Download missing inputs from Figshare and verify every input by MD5."""
    DATA.mkdir(parents=True, exist_ok=True)
    for filename, metadata in FIGSHARE_FILES.items():
        destination = DATA / filename
        expected = metadata["md5"]
        if destination.is_file():
            observed = file_md5(destination)
            if observed != expected:
                raise RuntimeError(
                    f"Checksum mismatch for {destination}: {observed} != {expected}. "
                    "Remove or rename the damaged file and rerun."
                )
            print(f"verified {filename}")
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"downloading {filename} from {metadata['url']}")
        try:
            with urllib.request.urlopen(metadata["url"]) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            observed = file_md5(temporary)
            if observed != expected:
                raise RuntimeError(
                    f"Checksum mismatch for downloaded {filename}: {observed} != {expected}"
                )
            temporary.replace(destination)
            print(f"verified {filename}")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def require_packages() -> None:
    required = {"scbiot": "scbiot", "scanpy": "scanpy", "moscot": "moscot", "wot": "wot", "POT": "ot"}
    missing = [package for package, module in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(
            "Missing benchmark packages: " + ", ".join(missing) +
            ". Install them before running the full benchmark."
        )


def as_dense(pi) -> np.ndarray:
    return pi.toarray() if sp.issparse(pi) else np.asarray(pi, dtype=np.float64)


def coupling_metrics(pi, labels_s: np.ndarray, labels_t: np.ndarray) -> dict[str, float]:
    pi = as_dense(pi)
    categories = np.unique(np.concatenate([labels_s, labels_t]))
    indices = {category: i for i, category in enumerate(categories)}
    ys = np.array([indices[value] for value in labels_s])
    yt = np.array([indices[value] for value in labels_t])
    onehot_s = np.zeros((len(ys), len(categories)))
    onehot_t = np.zeros((len(yt), len(categories)))
    onehot_s[np.arange(len(ys)), ys] = 1.0
    onehot_t[np.arange(len(yt)), yt] = 1.0

    col = pi.sum(0, keepdims=True)
    row = pi.sum(1, keepdims=True)
    col[col == 0] = 1.0
    row[row == 0] = 1.0
    pred_t = ((pi / col).T @ onehot_s).argmax(1)
    pred_s = ((pi / row) @ onehot_t).argmax(1)
    acc_fwd = float((pred_t == yt).mean())
    acc_bwd = float((pred_s == ys).mean())

    top1_fwd = float((labels_s[pi.argmax(0)] == labels_t).mean())
    top1_bwd = float((labels_t[pi.argmax(1)] == labels_s).mean())
    probability = pi / row
    entropy = -(probability * np.log(probability + 1e-12)).sum(1)
    entropy = float(np.mean(entropy) / np.log(probability.shape[1]))
    return {
        "acc_fwd": acc_fwd,
        "acc_bwd": acc_bwd,
        "acc_mean": 0.5 * (acc_fwd + acc_bwd),
        "top1_fwd": top1_fwd,
        "top1_bwd": top1_bwd,
        "top1_mean": 0.5 * (top1_fwd + top1_bwd),
        "entropy": entropy,
    }


def zscore_spatial(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(0)) / (values.std(0) + 1e-6)


def load_dataset(name: str, gpu: bool, seed: int):
    cfg = DATASETS[name]
    if cfg["type"] == "checkpoint":
        values = np.load(cfg["path"], allow_pickle=True)
        timepoint = values["timepoint"].astype(str)
        order = [
            stage for stage in
            ["E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]
            if stage in set(timepoint)
        ]
        return (
            values["X_ae"],
            values["spatial"],
            timepoint,
            values[cfg["label"]].astype(str),
            order,
            cfg["pairs"],
        )

    import scanpy as sc
    import scbiot as sb

    adata = sc.read_h5ad(cfg["path"])
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sb.pp.autoencoder(
        adata,
        input_key="counts",
        out_key="X_ae",
        batch_key="timepoint",
        n_top_genes=min(2000, adata.n_vars),
        l2=True,
        random_state=seed,
    )
    sb.integrate(
        adata,
        obsm_key="X_ae",
        batch_key="timepoint",
        out_key="X_scbiot_st",
        centroid=True,
        use_gpu=gpu,
        random_state=seed,
        verbose=False,
    )
    timepoint = adata.obs["timepoint"].astype(str).to_numpy()
    if hasattr(adata.obs["timepoint"], "cat"):
        order = list(adata.obs["timepoint"].cat.categories)
    else:
        order = list(dict.fromkeys(timepoint))
    pairs = list(zip(order[:-1], order[1:]))
    spatial = np.asarray(adata.obsm["spatial"]) if "spatial" in adata.obsm else None
    return (
        np.asarray(adata.obsm["X_ae"]),
        spatial,
        timepoint,
        adata.obs[cfg["label"]].astype(str).to_numpy(),
        order,
        pairs,
    )


def scbiot_coupling(source: np.ndarray, target: np.ndarray, gpu: bool):
    from scbiot.utils.ot_transport import compute_ot_alignment

    _, transport = compute_ot_alignment(
        source.astype(np.float32),
        target.astype(np.float32),
        use_gpu=gpu,
        transport_topk=64,
    )
    indices = np.asarray(transport["indices"], dtype=np.int64)
    weights = np.asarray(transport["weights"], dtype=np.float64)
    rows = np.repeat(np.arange(len(source)), indices.shape[1])
    return sp.csr_matrix(
        (weights.ravel(), (rows, indices.ravel())), shape=(len(source), len(target))
    )


def moscot_coupling(source: np.ndarray, target: np.ndarray):
    from moscot.problems.time import TemporalProblem

    values = np.vstack([source, target]).astype(np.float32)
    adata = ad.AnnData(X=values)
    adata.obs["day"] = np.r_[np.zeros(len(source)), np.ones(len(target))]
    adata.obsm["X_emb"] = values
    problem = TemporalProblem(adata).prepare(time_key="day", joint_attr="X_emb")
    problem = problem.solve(epsilon=EPSILON, scale_cost="mean")
    return np.asarray(problem[(0.0, 1.0)].solution.transport_matrix)


def moscot_spatial_coupling(
    source: np.ndarray, target: np.ndarray, spatial_s: np.ndarray, spatial_t: np.ndarray
):
    from moscot.problems.spatiotemporal import SpatioTemporalProblem

    values = np.vstack([source, target]).astype(np.float32)
    adata = ad.AnnData(X=values)
    adata.obs["day"] = np.r_[np.zeros(len(source)), np.ones(len(target))]
    adata.obsm["X_emb"] = values
    adata.obsm["spatial"] = np.vstack([spatial_s, spatial_t]).astype(np.float32)
    problem = SpatioTemporalProblem(adata).prepare(
        time_key="day", spatial_key="spatial", joint_attr="X_emb"
    )
    problem = problem.solve(alpha=0.5, epsilon=EPSILON, scale_cost="mean")
    return np.asarray(problem[(0.0, 1.0)].solution.transport_matrix)


def waddington_coupling(source: np.ndarray, target: np.ndarray):
    import wot

    values = np.vstack([source, target]).astype(np.float32)
    adata = ad.AnnData(X=values)
    adata.obs["day"] = np.r_[np.zeros(len(source)), np.ones(len(target))]
    model = wot.ot.OTModel(
        adata, epsilon=EPSILON, lambda1=1.0, lambda2=50.0, growth_iters=1
    )
    return np.asarray(model.compute_transport_map(0.0, 1.0).X)


def sinkhorn_coupling(source: np.ndarray, target: np.ndarray):
    import ot

    cost = ot.dist(source.astype(np.float64), target.astype(np.float64), metric="sqeuclidean")
    cost /= cost.max()
    mass_s = np.ones(len(source)) / len(source)
    mass_t = np.ones(len(target)) / len(target)
    return ot.sinkhorn(mass_s, mass_t, cost, reg=EPSILON)


def score(method: str, function, *args) -> dict[str, float]:
    labels_s, labels_t = args[-2], args[-1]
    started = time.perf_counter()
    coupling = function(*args[:-2])
    result = coupling_metrics(coupling, labels_s, labels_t)
    result["runtime_s"] = round(time.perf_counter() - started, 4)
    print(
        f"    {method:18s} top1={result['top1_mean']:.3f} "
        f"entropy={result['entropy']:.3f} runtime={result['runtime_s']:.3f}s"
    )
    return result


def run_benchmark(gpu: bool, n_per_stage: int, seed: int) -> dict:
    ensure_datasets()
    require_packages()
    rng = np.random.default_rng(seed)
    per_dataset: dict[str, dict[str, list[dict[str, float]]]] = {}

    for dataset in COMPUTE_ORDER:
        print(f"[{dataset}] loading and embedding")
        embedding, spatial, timepoint, labels, _order, pairs = load_dataset(dataset, gpu, seed)
        results: dict[str, list[dict[str, float]]] = {}
        for stage_s, stage_t in pairs:
            source_idx = np.where(timepoint == stage_s)[0]
            target_idx = np.where(timepoint == stage_t)[0]
            source_idx = rng.choice(source_idx, min(n_per_stage, len(source_idx)), replace=False)
            target_idx = rng.choice(target_idx, min(n_per_stage, len(target_idx)), replace=False)
            source = embedding[source_idx]
            target = embedding[target_idx]
            labels_s = labels[source_idx]
            labels_t = labels[target_idx]
            has_spatial = spatial is not None
            spatial_s = zscore_spatial(spatial[source_idx]) if has_spatial else None
            spatial_t = zscore_spatial(spatial[target_idx]) if has_spatial else None
            print(f"  {stage_s} -> {stage_t} ({len(source_idx)} x {len(target_idx)})")

            variants = {"scBIOT-noSpatial": (source, target)}
            if has_spatial:
                weight = float(np.mean(source.std(0))) * SPATIAL_W
                variants["scBIOT-ST"] = (
                    np.hstack([source, spatial_s * weight]),
                    np.hstack([target, spatial_t * weight]),
                )
            for method, (values_s, values_t) in variants.items():
                results.setdefault(method, []).append(
                    score(method, lambda a, b: scbiot_coupling(a, b, gpu), values_s, values_t,
                          labels_s, labels_t)
                )

            baseline_functions = {
                "moscot": moscot_coupling,
                "WaddingtonOT": waddington_coupling,
                "Sinkhorn-OT": sinkhorn_coupling,
            }
            if has_spatial:
                baseline_functions["moscot-spatial"] = (
                    lambda a, b: moscot_spatial_coupling(a, b, spatial_s, spatial_t)
                )
            for method, function in baseline_functions.items():
                results.setdefault(method, []).append(
                    score(method, function, source, target, labels_s, labels_t)
                )

        per_dataset[dataset] = results

    aggregate = {}
    for dataset, method_results in per_dataset.items():
        aggregate[dataset] = {}
        for method, records in method_results.items():
            aggregate[dataset][method] = {
                key: float(np.mean([record[key] for record in records])) for key in records[0]
            }
    return {
        "seed": seed,
        "n_per_stage": n_per_stage,
        "device": "gpu" if gpu else "cpu",
        "results": aggregate,
    }


def result_rows(payload: dict) -> list[dict]:
    rows = []
    results = payload["results"]
    for dataset in DISPLAY_ORDER:
        for method in METHODS:
            if method in results[dataset]:
                rows.append({"dataset": dataset, "method": method, **results[dataset][method]})
    return rows


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue LT Std", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(payload: dict, output: Path) -> None:
    set_plot_style()
    results = payload["results"]
    figure = plt.figure(figsize=(7.2, 2.45), constrained_layout=False)
    grid = figure.add_gridspec(
        1, 3, width_ratios=[2.65, 1.0, 1.0], left=0.07, right=0.985,
        top=0.76, bottom=0.27, wspace=0.43
    )
    axes = [figure.add_subplot(grid[0, i]) for i in range(3)]

    # a: sharpness-invariant top-1 label-transfer accuracy by dataset and method.
    axis = axes[0]
    bar_width = 0.28
    gap = 0.5
    cursor = 0.0
    centers = []
    seen = set()
    for dataset in DISPLAY_ORDER:
        present = [
            method for method in METHODS
            if method in results[dataset] and "top1_mean" in results[dataset][method]
        ]
        x_values = cursor + bar_width * np.arange(len(present))
        for x_value, method in zip(x_values, present):
            axis.bar(
                x_value,
                results[dataset][method]["top1_mean"],
                bar_width * 0.92,
                color=COLORS[method],
                label=method if method not in seen else None,
            )
            seen.add(method)
        centers.append(float(x_values.mean()))
        cursor = float(x_values[-1] + bar_width + gap)
    axis.set_xticks(centers, [DISPLAY_NAMES[name] for name in DISPLAY_ORDER], rotation=35, ha="right")
    axis.set_ylabel("Top-1 label-transfer accuracy")
    axis.set_ylim(0, 1.05)
    axis.axhline(0.5, linestyle=":", linewidth=0.5, color="0.5")
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
        columnspacing=0.9, handletextpad=0.35, handlelength=1.0
    )

    # b: mean normalized row entropy across datasets.
    axis = axes[1]
    entropy = [
        np.mean([results[name][method]["entropy"] for name in DISPLAY_ORDER if method in results[name]])
        for method in METHODS
    ]
    axis.bar(np.arange(len(METHODS)), entropy, color=[COLORS[method] for method in METHODS])
    axis.set_xticks(np.arange(len(METHODS)), METHODS, rotation=35, ha="right")
    axis.set_ylabel(r"Coupling entropy (mean) $\downarrow$")

    # c: mean recorded runtime per adjacent-stage pair, shown on a log scale.
    axis = axes[2]
    runtime = np.array([
        np.mean([results[name][method]["runtime_s"] for name in DISPLAY_ORDER if method in results[name]])
        for method in METHODS
    ])
    floor = float(runtime.min()) / 5.0
    axis.set_yscale("log")
    axis.set_ylim(floor, float(runtime.max()) * 1.6)
    axis.bar(
        np.arange(len(METHODS)), runtime - floor, bottom=floor,
        color=[COLORS[method] for method in METHODS]
    )
    axis.set_xticks(np.arange(len(METHODS)), METHODS, rotation=35, ha="right")
    axis.set_ylabel(r"Runtime (s) per pair (mean) $\downarrow$")

    for label, axis in zip("abc", axes):
        position = axis.get_position()
        figure.text(
            position.x0 - 0.025, position.y1 + 0.015, label,
            fontsize=12, fontweight="bold", ha="left", va="bottom"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=2 / 25.4)
    figure.savefig(output.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=2 / 25.4)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.download_only:
        ensure_datasets()
        print(f"datasets ready from {FIGSHARE_ARTICLE_URL}")
        return
    if args.results_json:
        with args.results_json.open() as handle:
            payload = json.load(handle)
    else:
        gpu = use_gpu(args.device)
        print(f"Running benchmark on {'GPU' if gpu else 'CPU'} with seed {args.seed}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            payload = run_benchmark(gpu, args.n_per_stage, args.seed)
        result_json = args.output.with_name(args.output.name + "_results").with_suffix(".json")
        with result_json.open("w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {result_json}")

    table = pd.DataFrame(result_rows(payload))
    table_path = args.output.with_name(args.output.name + "_source_data").with_suffix(".csv")
    table.to_csv(table_path, index=False)
    plot(payload, args.output)
    print(f"wrote {args.output.with_suffix('.pdf')}")
    print(f"wrote {args.output.with_suffix('.png')}")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
