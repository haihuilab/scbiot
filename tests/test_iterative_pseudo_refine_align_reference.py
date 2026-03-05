import numpy as np
import pandas as pd


class _DummyAdata:
    def __init__(self, X: np.ndarray, batches: list[str], labels: list[object]):
        self.obsm = {"X_pca": np.asarray(X, dtype=np.float32)}
        self.obs = pd.DataFrame({"batch": batches, "label": labels})
        self.uns = {}
        self.obs_names = pd.Index([f"cell{i}" for i in range(len(batches))])

    @property
    def n_obs(self) -> int:
        return int(self.obs.shape[0])


def _resolve_query_mask(labels: pd.Series, unlabeled_category):
    if unlabeled_category is None:
        return labels.isna()
    if isinstance(unlabeled_category, (list, tuple, set)):
        return labels.isna() | labels.isin(unlabeled_category)
    return labels.isna() | labels.eq(unlabeled_category)


def test_iterative_pseudo_refine_align_reference_runs_and_stops_early(monkeypatch):
    import importlib

    mod = importlib.import_module("scbiot.ot.iterative_refine")

    X = np.array(
        [
            [0.0, 0.0],   # ref
            [0.0, 1.0],   # ref
            [1.0, 0.0],   # ref
            [10.0, 10.0], # query
            [10.0, 11.0], # query
            [11.0, 10.0], # query
        ],
        dtype=np.float32,
    )
    batches = ["A", "B", "A", "B", "A", "B"]
    labels = ["type1", "type1", "type1", "Unknown", "Unknown", "Unknown"]
    adata = _DummyAdata(X, batches=batches, labels=labels)
    labels_before = adata.obs["label"].copy()

    integrate_calls: list[tuple[str, str, object, bool]] = []

    def _fake_integrate_ot(adata, *, obsm_key, out_key, prealign, align_reference, **kwargs):
        integrate_calls.append((obsm_key, out_key, prealign, bool(align_reference)))
        adata.obsm[out_key] = np.asarray(adata.obsm[obsm_key], dtype=np.float32, order="C").copy()
        return adata, {"ok": True}

    def _fake_predict_pseudo_labels(adata, *, label_key, unlabeled_category, **kwargs):
        n = int(adata.n_obs)
        q = _resolve_query_mask(adata.obs[label_key], unlabeled_category).to_numpy()
        pred_lab = np.full(n, None, dtype=object)
        pred_conf = np.full(n, np.nan, dtype=np.float32)
        pred_lab[q] = "type1"
        pred_conf[q] = 0.90
        return pred_lab, pred_conf

    monkeypatch.setattr(mod, "integrate_ot", _fake_integrate_ot)
    monkeypatch.setattr(mod, "predict_pseudo_labels", _fake_predict_pseudo_labels)

    adata, metrics = mod.iterative_pseudo_refine_align_reference(
        adata,
        obsm_key="X_pca",
        batch_key="batch",
        out_key="X_align_ref",
        label_key="label",
        unlabeled_category=("Unknown",),
        max_rounds=3,
        min_conf=0.80,
        ema=0.30,
        min_count=1,
        verbose=False,
    )

    assert adata.obs["label"].equals(labels_before)
    assert "pseudo_label" in adata.obs
    assert "pseudo_label_conf" in adata.obs

    q = _resolve_query_mask(adata.obs["label"], ("Unknown",)).to_numpy()
    assert (adata.obs["pseudo_label"].to_numpy()[q] == "type1").all()
    assert np.allclose(adata.obs["pseudo_label_conf"].to_numpy(dtype=np.float32)[q], 0.90, atol=1e-6)

    # Round 0: uses input embedding, with coral. Round 1: feeds back out_key, no prealign.
    assert integrate_calls[0][0] == "X_pca"
    assert integrate_calls[0][1] == "X_align_ref"
    assert integrate_calls[0][2] == "coral"
    assert integrate_calls[0][3] is True

    assert integrate_calls[1][0] == "X_align_ref"
    assert integrate_calls[1][1] == "X_align_ref"
    assert integrate_calls[1][2] is None
    assert integrate_calls[1][3] is True

    # First round has no "prev ok" cells so change_rate stays 1; second round stabilizes and stops.
    assert len(metrics) == 2
    assert metrics[0]["round"] == 0
    assert metrics[1]["round"] == 1
    assert metrics[1]["change_rate"] == 0.0


def test_iterative_pseudo_refine_align_reference_requires_correct_unlabeled_category():
    import importlib

    mod = importlib.import_module("scbiot.ot.iterative_refine")

    X = np.eye(4, dtype=np.float32)
    batches = ["A", "A", "B", "B"]
    labels = ["type1", "type1", "Unknown", "Unknown"]
    adata = _DummyAdata(X, batches=batches, labels=labels)

    # "Unknown" is not equal to "unknown" with the current exact-match mask semantics.
    try:
        mod.iterative_pseudo_refine_align_reference(
            adata,
            obsm_key="X_pca",
            batch_key="batch",
            out_key="X_align_ref",
            label_key="label",
            unlabeled_category="unknown",
            max_rounds=1,
            verbose=False,
        )
    except ValueError as exc:
        assert "Reference/query subset empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty query subset")

