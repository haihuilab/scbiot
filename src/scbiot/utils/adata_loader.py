# ============== paste-and-run: loaders_from_adata_quantile_split.py ==============
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader

# optional: sklearn for split + quantile transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# -------- helpers --------
def _ensure_dense_float32(X):
    """Return dense float32 array from (possibly) sparse matrix."""
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
    except Exception:
        pass
    return np.asarray(X, dtype=np.float32)

def _encode_by_count(series: pd.Series) -> Tuple[np.ndarray, Dict, Dict]:
    """Encode categories -> ints by descending frequency."""
    s = series.astype("string").fillna("<UNK>")
    counts = s.value_counts(dropna=False)
    ordered = list(counts.index)
    to_id = {lab: i for i, lab in enumerate(ordered)}
    codes = s.map(to_id).to_numpy(dtype=np.int64)
    to_label = {v: k for k, v in to_id.items()}
    return codes, to_id, to_label

def _get_numeric_matrix(adata, X: Optional[str]):
    """
    If X is a key in adata.obsm, use it; if X in (None, 'X'), use adata.X.
    """
    if isinstance(X, str) and X not in (None, "X"):
        if X not in adata.obsm:
            raise KeyError(f"'{X}' not found in adata.obsm")
        return _ensure_dense_float32(adata.obsm[X])
    return _ensure_dense_float32(adata.X)

def _fit_quantile_on_train_transform_both(
    X_train: np.ndarray,
    X_test: np.ndarray,
    output_distribution: str = "normal",
    n_quantiles: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, QuantileTransformer]:
    """
    Fit QuantileTransformer on train only; transform both train/test.
    Gracefully handle constant columns (leave them as-is).
    """
    # Identify constant columns to skip
    ptp = np.ptp(X_train, axis=0)
    const_mask = (ptp == 0.0)

    # Work on the varying columns
    var_cols = np.where(~const_mask)[0]
    Xtr = X_train.copy()
    Xte = X_test.copy()

    if len(var_cols) > 0:
        qt = QuantileTransformer(
            n_quantiles=min(n_quantiles, X_train.shape[0]),
            output_distribution=output_distribution,
            subsample=int(1e5),
            copy=True,
            random_state=random_state,
        )
        Xtr[:, var_cols] = qt.fit_transform(X_train[:, var_cols])
        Xte[:, var_cols] = qt.transform(X_test[:, var_cols])
    else:
        qt = None  # all columns constant

    return Xtr, Xte, qt

# -------- arrays (no barcodes) --------
def _arrays_from_adata(
    adata,
    X: str = "scBIOT_OT",
    batch: str = "batch",
    pseudo_labels: str = "scBIOT_leiden",
    true_labels: str = "cell_type",
):
    """
    Build arrays directly from AnnData without saving anything.
    Returns:
      X_num: float32 [N, D]
      X_cat: int64   [N, 3]  columns = [batch, pseudo_labels, true_labels]
      mappings: dict for each cat
    """
    X_num = _get_numeric_matrix(adata, X)

    cat_keys = [("batch", batch),
                ("pseudo_labels", pseudo_labels),
                ("true_labels", true_labels)]

    cats: List[np.ndarray] = []
    mappings: Dict[str, Dict[str, Dict]] = {}

    for out_name, obs_key in cat_keys:
        if obs_key not in adata.obs.columns:
            raise KeyError(f"'{obs_key}' not found in adata.obs")
        codes, to_id, to_label = _encode_by_count(adata.obs[obs_key])
        cats.append(codes)
        mappings[out_name] = {"to_id": to_id, "to_label": to_label}

    X_cat = np.stack(cats, axis=1).astype(np.int64)
    return X_num, X_cat, mappings

# -------- dataset (no barcodes) --------
class SCDataset(Dataset):
    """
    Yields:
      (x_num, pre_x_num, x_cat)               if track_indices=False
      (idx, x_num, pre_x_num, x_cat)          if track_indices=True
    """
    def __init__(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        to_torch: bool = True,
        device: Optional[str] = None,
        track_indices: bool = False,
    ):
        self.X_num = X_num.astype(np.float32, copy=False)
        self.pre_X_num = np.empty((self.X_num.shape[0], 0), dtype=np.float32)  # empty 2nd view
        self.X_cat = X_cat.astype(np.int64, copy=False)
        self.to_torch = to_torch
        self.device = torch.device(device) if device else torch.device("cpu")
        self.track_indices = track_indices

        if self.to_torch:
            self._X_num = torch.from_numpy(self.X_num).to(self.device)
            self._pre_X_num = torch.from_numpy(self.pre_X_num).to(self.device)
            self._X_cat = torch.from_numpy(self.X_cat).to(self.device)

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        if self.to_torch:
            tup = (self._X_num[idx], self._pre_X_num[idx], self._X_cat[idx])
        else:
            tup = (self.X_num[idx], self.pre_X_num[idx], self.X_cat[idx])
        if self.track_indices:
            return (idx, *tup)
        return tup

# -------- public API: build loaders with quantile norm + split --------
# === drop-in replacement ===
from typing import Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# --- minimal imports (adjust if you already have these elsewhere) ---
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from scipy import sparse

# ------------------------- helpers -------------------------
def _get_matrix_from_adata(adata, X: Optional[str]):
    """
    Returns a dense float32 (N, D) matrix from AnnData.
    Tries obsm[X], then layers[X], else adata.X when X in {None,"X"}.
    """
    arr = None
    if isinstance(X, str) and X in getattr(adata, "obsm", {}):
        arr = adata.obsm[X]
    elif isinstance(X, str) and hasattr(adata, "layers") and X in adata.layers:
        arr = adata.layers[X]
    elif X is None or X == "X":
        arr = adata.X
    else:
        # final fallback to adata.X if key not found
        arr = adata.X

    if sparse.issparse(arr):
        arr = arr.toarray()
    arr = np.asarray(arr, dtype=np.float32)
    return arr

def _safe_encode_obs_column(adata, colname: Optional[str]):
    """
    Encode an adata.obs column to integer codes [0..K-1].
    Returns (codes:int64[N], categories:list[str]) or (None, None) if col is None/missing.
    NaNs are mapped to a '__NA__' category to avoid -1 codes.
    """
    if colname is None:
        return None, None
    if colname not in adata.obs.columns:
        return None, None

    s = adata.obs[colname]
    # Ensure categorical with explicit NA bucket if needed
    if s.isna().any():
        s = s.fillna("__NA__")
    s = s.astype("category")
    cats = list(s.cat.categories)
    codes = s.cat.codes.to_numpy(dtype=np.int64)
    return codes, cats

def _arrays_from_adata(adata, X: str, batch: str,
                       pseudo_labels: Optional[str], true_labels: Optional[str]):
    """
    Builds numerical matrix X_num_all and stacked categorical matrix X_cat_all
    with columns in order: [batch] + [pseudo_labels?] + [true_labels?] (only those that exist).
    Returns (X_num_all:float32[N,D], X_cat_all:int64[N,C], mappings:dict)
    """
    X_num_all = _get_matrix_from_adata(adata, X)

    cat_cols = []
    mappings = {}

    # batch (required in your pipeline; if missing, we still proceed without it)
    batch_codes, batch_cats = _safe_encode_obs_column(adata, batch)
    if batch_codes is not None:
        cat_cols.append(batch_codes)
        mappings["batch"] = batch_cats

    # pseudo labels (optional)
    pseudo_codes, pseudo_cats = _safe_encode_obs_column(adata, pseudo_labels)
    if pseudo_codes is not None:
        cat_cols.append(pseudo_codes)
        mappings[pseudo_labels] = pseudo_cats  # key with the actual column name

    # true labels (optional)
    true_codes, true_cats = _safe_encode_obs_column(adata, true_labels)
    if true_codes is not None:
        cat_cols.append(true_codes)
        mappings[true_labels] = true_cats  # key with the actual column name

    if len(cat_cols):
        X_cat_all = np.vstack(cat_cols).T  # shape (N, C)
    else:
        N = X_num_all.shape[0]
        X_cat_all = np.zeros((N, 0), dtype=np.int64)

    return X_num_all, X_cat_all, mappings

# ------------------------- main function -------------------------
def build_loaders_from_adata(
    adata,
    X: str = "scBIOT_OT",
    batch_labels: str = "batch",
    pseudo_labels: Optional[str] = "scBIOT_leiden",
    true_labels: Optional[str] = "cell_type",
    batch_size: int = 2048,
    device: Optional[str] = None,
    num_workers: int = 0,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    test_size: float = 0.1,
    random_state: int = 42,
    quantile_norm: bool = True,
    output_distribution: str = "normal",  # kept for API compatibility; ignored when legacy norm is used
    n_quantiles: int = 1000,              # kept for API compatibility; ignored when legacy norm is used
    track_indices: bool = False,
):
    """
    Returns:
      train:   (train_ds, train_loader)
      test:    (test_ds,  test_loader)
      extras:  {
        "mappings":..., "categories":[sizes], "d_numerical":D,
        "qt": QuantileTransformer|None,
        "split_idx": (train_idx, test_idx)
      }
    """
    # 1) pull arrays & encode cats (now robust to None / missing columns)
    X_num_all, X_cat_all, mappings = _arrays_from_adata(
        adata, X=X, batch=batch_labels, pseudo_labels=pseudo_labels, true_labels=true_labels
    )
    N, D = X_num_all.shape

    # 2) split indices (no stratify to replicate legacy)
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=True, stratify=None
    )

    # 3) split arrays
    Xtr_num, Xte_num = X_num_all[train_idx], X_num_all[test_idx]
    Xtr_cat, Xte_cat = X_cat_all[train_idx], X_cat_all[test_idx]

    # 4) LEGACY quantile normalization (fit on train, transform both)
    qt = None
    if quantile_norm:
        # Legacy choices (match your previous behavior)
        n_train = Xtr_num.shape[0]
        legacy_nq = max(min(n_train // 30, 1000), 10)
        qt = QuantileTransformer(
            output_distribution=output_distribution,
            n_quantiles=legacy_nq,
            subsample=int(1e9),
            random_state=random_state,
        )
        qt.fit(Xtr_num)
        Xtr_num = qt.transform(Xtr_num)
        Xte_num = qt.transform(Xte_num)

    # 5) category sizes (computed from the actual train slice; empty if no cats)
    if Xtr_cat.shape[1] == 0:
        categories = []
    else:
        categories = [int(Xtr_cat[:, i].max()) + 1 for i in range(Xtr_cat.shape[1])]

    # 6) datasets/loaders (assumes your SCDataset signature: (X_num, X_cat, ...))
    train_ds = SCDataset(Xtr_num, Xtr_cat, to_torch=True, device=device, track_indices=track_indices)
    test_ds  = SCDataset(Xte_num, Xte_cat, to_torch=True, device=device, track_indices=track_indices)

    pin = (device is not None) and ("cuda" in str(device).lower())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers, pin_memory=pin
    )

    extras = {
        "mappings": mappings,          # dict of {<obs_col>: [categories]}, only for columns that existed
        "categories": categories,      # list[int], one per categorical column actually used
        "d_numerical": int(D),
        "qt": qt,                      # QuantileTransformer or None
        "split_idx": (train_idx, test_idx),
    }
    return (train_ds, train_loader), (test_ds, test_loader), extras
