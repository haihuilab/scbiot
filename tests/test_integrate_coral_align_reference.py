import numpy as np
import pandas as pd


class _DummyAdata:
    def __init__(self, X: np.ndarray, batches: list[str], labels: list[object]):
        self.obsm = {"X_pca": np.asarray(X, dtype=np.float32)}
        self.obs = pd.DataFrame({"batch": batches, "label": labels})
        self.uns = {}
        self.obs_names = pd.Index([f"cell{i}" for i in range(len(batches))])


def test_align_reference_coral_uses_ref_query_masks_and_keeps_reference_fixed(monkeypatch):
    import importlib

    integrate_mod = importlib.import_module("scbiot.ot.integrate")

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
    labels = ["type1", "type1", "type1", "unknown", "unknown", "unknown"]
    adata = _DummyAdata(X, batches=batches, labels=labels)

    def _fake_compute_ot_alignment(query_emb, ref_emb, **kwargs):
        aligned = np.asarray(query_emb, dtype=np.float32, order="C").copy()
        n = aligned.shape[0]
        return aligned, {
            "indices": np.zeros((n, 1), dtype=np.int32),
            "weights": np.ones((n, 1), dtype=np.float32),
            "residual": None,
        }

    monkeypatch.setattr(integrate_mod, "compute_ot_alignment", _fake_compute_ot_alignment)

    integrate_mod.integrate_ot(
        adata,
        obsm_key="X_pca",
        batch_key="batch",
        out_key="scBIOT",
        label_key="label",
        unlabeled_category="unknown",
        align_reference=True,
        prealign="coral",
        use_gpu=False,
        verbose=False,
        modality="rna",
        random_state=0,
    )

    out = adata.obsm["scBIOT"]
    ref_mask = ~adata.obs["label"].eq("unknown")
    query_mask = ~ref_mask

    assert np.array_equal(out[ref_mask.to_numpy()], X[ref_mask.to_numpy()])
    assert not np.array_equal(out[query_mask.to_numpy()], X[query_mask.to_numpy()])
