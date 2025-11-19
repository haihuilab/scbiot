import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, math, time, warnings
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


# ============================ Prototype head & helpers ============================
class ProtoHead(nn.Module):
    """
    Learnable prototypes for clustering on latent z. Cosine logits with temperature.
    """
    def __init__(self, d: int, K: int, tau: float = 0.07, cosine: bool = True, ema_m: float = 0.95):
        super().__init__()
        self.C = nn.Parameter(torch.randn(K, d))
        self.tau = tau
        self.cosine = cosine
        self.ema_m = ema_m
        nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        if self.cosine:
            z = F.normalize(z, dim=1)
            C = F.normalize(self.C, dim=1)
            return (z @ C.t()) / max(self.tau, 1e-6)
        else:
            z2 = (z ** 2).sum(1, keepdim=True)
            C2 = (self.C ** 2).sum(1, keepdim=True).t()
            return -(z2 - 2 * z @ self.C.t() + C2) / (2 * (self.tau ** 2) + 1e-8)

    @torch.no_grad()
    def init_from_batch(self, z: torch.Tensor):
        B = z.size(0)
        K = self.C.size(0)
        idx = torch.randperm(B, device=z.device)[:K]
        C0 = z[idx]
        if self.cosine:
            C0 = F.normalize(C0, dim=1)
        self.C.data.copy_(C0)

    @torch.no_grad()
    def ema_update(self, z: torch.Tensor, p: torch.Tensor):
        """
        EMA update of prototypes using soft assignments p. Works with cosine or L2.
        z: [B,D], p: [B,K]
        """
        if self.cosine:
            z = F.normalize(z, dim=1)
            C_new = (p.t() @ z) / (p.sum(0, keepdim=True).t() + 1e-8)
            C_new = F.normalize(C_new, dim=1)
        else:
            C_new = (p.t() @ z) / (p.sum(0, keepdim=True).t() + 1e-8)
        self.C.data = self.ema_m * self.C.data + (1.0 - self.ema_m) * C_new

def dec_targets(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """DEC sharpened, balanced targets from softmax q."""
    fk = q.sum(0, keepdim=True) + eps
    p = (q ** 2) / fk
    p = p / (p.sum(1, keepdim=True) + eps)
    return p.detach()

def proto_center_loss(z: torch.Tensor, proto_head: ProtoHead, p: torch.Tensor) -> torch.Tensor:
    """Soft center loss: pull z towards prototypes weighted by p."""
    if proto_head.cosine:
        z_n = F.normalize(z, dim=1)
        C = F.normalize(proto_head.C, dim=1)
    else:
        z_n = z
        C = proto_head.C
    diff = z_n.unsqueeze(1) - C.unsqueeze(0)          # [B,K,D]
    return (diff.pow(2) * p.unsqueeze(-1)).sum(dim=(1, 2)).mean()

def prototype_repulsion(proto_head: ProtoHead, target: float = 0.0) -> torch.Tensor:
    """
    Push prototypes apart: hinge on pairwise cosine similarity.
    target: desired max cosine(sim) between different prototypes.
    """
    C = F.normalize(proto_head.C, dim=1) if proto_head.cosine else proto_head.C
    S = C @ C.t()                                   # [K,K]
    mask = ~torch.eye(S.size(0), dtype=torch.bool, device=S.device)
    S_ij = S[mask]
    return F.relu(S_ij - target).pow(2).mean()

def cosface_margin_logits(logits: torch.Tensor, y: torch.Tensor, m: float = 0.15) -> torch.Tensor:
    """
    CosFace-style additive cosine margin: subtract m from the target logit.
    logits: [B,K], y: [B] hard labels
    """
    B, K = logits.shape
    logits_m = logits.clone()
    logits_m[torch.arange(B, device=logits.device), y] -= m
    return logits_m

def sym_kl(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two categorical distributions."""
    q1 = q1.clamp_min(1e-8)
    q2 = q2.clamp_min(1e-8)
    return 0.5 * (F.kl_div(q1.log(), q2, reduction='batchmean') +
                  F.kl_div(q2.log(), q1, reduction='batchmean'))

def smooth_ce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 0.1):
    """Label-smoothed cross-entropy."""
    if logits is None:
        return logits.new_tensor(0.0)
    C = logits.size(-1)
    with torch.no_grad():
        y = torch.zeros_like(logits).scatter_(1, target.view(-1, 1), 1.0)
        y = (1.0 - eps) * y + eps / float(C)
    logp = F.log_softmax(logits, dim=-1)
    return -(y * logp).sum(dim=-1).mean()


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z, label_eps: float = 0.1):
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0.0
    acc = 0.0
    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


