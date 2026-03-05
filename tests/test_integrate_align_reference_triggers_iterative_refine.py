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


def test_integrate_align_reference_calls_iterative_refine(monkeypatch):
    import scbiot as scb

    X = np.eye(4, dtype=np.float32)
    adata = _DummyAdata(
        X,
        batches=["A", "A", "B", "B"],
        labels=["type1", "type1", "Unknown", "Unknown"],
    )

    called = {"ok": False}

    def _fake_iterative(adata_in, **kwargs):
        called["ok"] = True
        # basic sanity on kwargs passed through
        assert kwargs["align_reference"] if "align_reference" in kwargs else True
        assert kwargs["label_key"] == "label"
        assert kwargs["unlabeled_category"] == "Unknown"
        assert kwargs["out_key"] == "X_out"
        adata_in.obsm["X_out"] = np.asarray(adata_in.obsm["X_pca"], dtype=np.float32).copy()
        rounds = [{"round": 0, "integrate": {"mix": 0.0, "it": 0}, "change_rate": 0.0, "n_pseudo": 2, "n_query": 2}]
        adata_in.obs["pseudo_label"] = adata_in.obs["label"].astype(object)
        adata_in.obs["pseudo_label_conf"] = np.ones(adata_in.n_obs, dtype=np.float32)
        return adata_in, rounds

    # Ensure the wrapper calls iterative refinement and does not fall back to integrate_ot directly.
    monkeypatch.setattr(scb.ot, "iterative_pseudo_refine_align_reference", _fake_iterative)

    adata_out, metrics = scb.ot.integrate(
        adata,
        obsm_key="X_pca",
        batch_key="batch",
        out_key="X_out",
        label_key="label",
        unlabeled_category="Unknown",
        align_reference=True,
        verbose=False,
        use_gpu=False,
    )

    assert called["ok"] is True
    assert "rounds" in metrics
    assert adata_out is adata
    assert "X_out" in adata_out.obsm

