import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, math, time, warnings
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


# ----------------------------- helpers: explicit label extraction -----------------------------
def _codes_from_obs(adata: "sc.AnnData", key: Optional[str]) -> Optional[torch.Tensor]:
    """
    Return global integer codes for adata.obs[key] across ALL cells (consistent train/test).
    Missing key -> None. NaNs map to code -1.
    """
    if key is None or key not in adata.obs.columns:
        return None
    cats = pd.Categorical(adata.obs[key])
    codes = np.asarray(cats.codes, dtype=np.int64)  # -1 for NaN
    return torch.from_numpy(codes)

def _split_by_idx(codes: Optional[torch.Tensor], tr_idx, te_idx) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if codes is None:
        return None, None
    return codes[tr_idx].clone(), codes[te_idx].clone()

def _ensure_tensor(x, fill_value: int = -1) -> torch.Tensor:
    """
    Ensure we always get a 1-D LongTensor (even if x is None).
    """
    if x is None:
        return torch.empty(0, dtype=torch.long)
    return x.to(torch.long).view(-1)


# ----------------------------- dataset: two numerical views + explicit labels -----------------------------
class SCDataset2View(Dataset):
    """
    Yields:
      x_num, pre_x_num, batch_lab, pseudo_lab, true_lab
    All labels are 1-D LongTensors. If a label stream wasn't provided, it is a tensor of shape [N] filled with -1.
    """
    def __init__(self, X_num: torch.Tensor, pre_X_num: torch.Tensor,
                 batch_lab: torch.Tensor, pseudo_lab: torch.Tensor, true_lab: torch.Tensor):
        super().__init__()
        self.X_num = X_num
        self.pre_X_num = pre_X_num
        self.batch_lab = batch_lab
        self.pseudo_lab = pseudo_lab
        self.true_lab = true_lab

    def __len__(self): 
        return self.X_num.size(0)

    def __getitem__(self, idx):
        return (self.X_num[idx],
                self.pre_X_num[idx],
                self.batch_lab[idx],
                self.pseudo_lab[idx],
                self.true_lab[idx])
