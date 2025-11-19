import os, math, time, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import argparse
# import wandb

warnings.filterwarnings('ignore')

from .vae import Model_VAE, Encoder_model, Decoder_model
from ..utils.adata_loader import build_loaders_from_adata
from ..pp.setup_anndata import get_anndata_setup
from ..utils.train_utils import (
    assign_soft_cluster,
    get_cosine_schedule_with_warmup, 
    norm_batch, 
    AverageMeter, 
    split_batch_precoder, 
    UniformLabelSampler, 
    set_seed, 
    calculate_metrics
)
from ..utils.helpers import (
    _codes_from_obs, 
    _split_by_idx, 
    SCDataset2View,
    _ensure_tensor
    
)
from ..utils.losses import (
    ProtoHead, 
    compute_loss, 
    cosface_margin_logits,
    smooth_ce_loss,
    sym_kl,
    dec_targets,
    proto_center_loss,
    prototype_repulsion
)




# ============================== training hyperparams ==============================
LR = 1e-2
WD = 0
D_TOKEN = 4
N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2

BATCH_SIZE = 8192
NUM_EPOCHS = 80


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

# ---------------------- configs ---------------------- #
@dataclass
class HyperParams:
    proto_tau: float      = 0.08
    lam_clust: float      = 1.0
    lam_center: float     = 1.0
    lam_sup: float        = 1.0   
    proto_ema_m: float    = 0.95 
    label_eps: float      = 0.1   
    lam_margin: float     = 1.5
    cosface_m: float      = 0.15
    conf_thr: float       = 0.8
    lam_repulse: float    = 1.0
    repulse_target: float = 0.1
    lam_cons: float       = 1.5


# ========================================= Main =========================================
class VAEModel:
    def __init__(
        self,
        adata,
        num_layers: Optional[int] = NUM_LAYERS,
        # d_numerical: Optional[int] = None,
        categories: Optional[Sequence[int]] = None,
        d_token: Optional[int] = D_TOKEN,
        *,
        n_head: int = N_HEAD,
        factor: int = FACTOR,
        bias: bool = True,
        var_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        pseudo_key: Optional[str] = None,
        true_key: Optional[str] = None,
        threshold: float = 0.1,
        num_clusters: int = 64,
        random_seed: int = 42,
        training_steps: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
        lr: float = LR,
        num_epochs: int = NUM_EPOCHS,
        hyperparams: Optional[HyperParams] = None,
        prior_pcr: float = 1.0,
        verbose: bool = False,
    ):
        self.adata = adata
        self.num_layers = num_layers or NUM_LAYERS
        self.d_token = d_token or D_TOKEN
        self.n_head = n_head
        self.factor = factor
        self.bias = bias
        # self.d_numerical = d_numerical
        # self.categories = categories
        self.hyperparams = hyperparams or HyperParams()
        self.threshold = float(threshold)
        self.num_clusters = int(num_clusters)
        self.random_seed = int(random_seed)
        self.training_steps = training_steps
        # ensure scalar hyperparameters even if tuples/arrays slip in from config
        self.batch_size = int(batch_size) if batch_size is not None else int(BATCH_SIZE)
        self.lr = float(lr) if lr is not None else float(LR)
        self.num_epochs = int(num_epochs) if num_epochs is not None else int(NUM_EPOCHS)
        self.device = _resolve_device(device)
        self.device_is_cuda = ('cuda' in str(self.device)) and torch.cuda.is_available()
        self.verbose = bool(verbose)
        self.prior_pcr = float(prior_pcr)
        set_seed(self.random_seed)

        # ===== proto / separation hyperparams =====
        self.proto_tau     = float(getattr(self.hyperparams, "proto_tau", 0.08))
        self.lam_clust     = float(getattr(self.hyperparams, "lam_clust", 1.0))
        self.lam_center    = float(getattr(self.hyperparams, "lam_center", 0.5))
        self.lam_sup_max   = float(getattr(self.hyperparams, "lam_sup", 0.2))
        self.label_eps     = float(getattr(self.hyperparams, "label_eps", 0.1))
        self.proto_ema_m   = float(getattr(self.hyperparams, "proto_ema_m", 0.95))
        self.lam_margin    = float(getattr(self.hyperparams, "lam_margin", 0.6))
        self.cosface_m     = float(getattr(self.hyperparams, "cosface_m", 0.2))
        self.conf_thr      = float(getattr(self.hyperparams, "conf_thr", 0.6))
        self.lam_repulse   = float(getattr(self.hyperparams, "lam_repulse", 0.1))
        self.repulse_tar   = float(getattr(self.hyperparams, "repulse_target", 0.0))
        self.lam_cons      = float(getattr(self.hyperparams, "lam_cons", 0.2))

        self.curr_dir = Path.cwd()
        self.ckpt_dir = f'{self.curr_dir}/scbiot_ckpt/'
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.model_save_path = f'{self.ckpt_dir}/model.pt'
        self.encoder_save_path = f'{self.ckpt_dir}/encoder.pt'
        self._latents_cache: Optional[torch.Tensor] = None

        registry = get_anndata_setup(adata)
        self.var_key = var_key or registry.get("var_key") or "scBIOT_OT"
        self.batch_key = batch_key or registry.get("batch_key") or "batch"

        default_pseudo = registry.get("pseudo_key")
        if default_pseudo is None and "leiden_scBIOT_OT" in adata.obs.columns:
            default_pseudo = "leiden_scBIOT_OT"
        default_true = registry.get("true_key")
        if default_true is None and "cell_type" in adata.obs.columns:
            default_true = "cell_type"

        self.pseudo_key = pseudo_key if pseudo_key is not None else default_pseudo
        if self.pseudo_key is not None and self.pseudo_key not in adata.obs.columns:
            warnings.warn(f"'{self.pseudo_key}' not found in adata.obs; disabling pseudo labels.")
            self.pseudo_key = None

        self.true_key = true_key if true_key is not None else default_true
        if self.true_key is not None and self.true_key not in adata.obs.columns:
            warnings.warn(f"'{self.true_key}' not found in adata.obs; disabling true labels.")
            self.true_key = None

        if self.var_key is None:
            raise ValueError("var_key must be provided or registered via setup_anndata.")
        if self.batch_key is None:
            raise ValueError("batch_key must be provided or registered via setup_anndata.")

        (train_ds0, _), (test_ds0, _), extras = build_loaders_from_adata(
            self.adata,
            X=self.var_key,
            batch_labels=self.batch_key,
            pseudo_labels=self.pseudo_key,
            true_labels=self.true_key,
            batch_size=self.batch_size,     # loader returned here is unused; we rebuild with 2nd view
            device=None,
            test_size=0.1,
            quantile_norm=True,
            output_distribution="normal"
        )
      
        self.d_numerical = int(extras["d_numerical"])
        self.categories = extras["categories"]
        self.split_idx  = extras.get("split_idx", None)        

        # ===== tensors (from ds0) =====
        Xtr_num = torch.as_tensor(train_ds0.X_num, dtype=torch.float32)
        Xte_num = torch.as_tensor(test_ds0.X_num,  dtype=torch.float32)
        
        # We'll compute batch labels from adata (explicit) before norm.
        # Extract global codes for each label role.
        batch_codes_all  = _codes_from_obs(self.adata, self.batch_key)
        pseudo_codes_all = _codes_from_obs(self.adata, self.pseudo_key)
        true_codes_all   = _codes_from_obs(self.adata, self.true_key)

        if self.split_idx is None:
            # Fallback: assume first N_train rows are train (not ideal, but keeps script running)
            n_train = Xtr_num.shape[0]
            tr_idx = np.arange(n_train, dtype=np.int64)
            te_idx = np.arange(n_train, n_train + Xte_num.shape[0], dtype=np.int64)
        else:
            tr_idx, te_idx = self.split_idx

        # Split label codes into train/test and ensure 1-D LongTensors
        batch_tr, batch_te   = _split_by_idx(batch_codes_all,  tr_idx, te_idx)
        pseudo_tr, pseudo_te = _split_by_idx(pseudo_codes_all, tr_idx, te_idx)
        true_tr, true_te     = _split_by_idx(true_codes_all,   tr_idx, te_idx)

        # Presence flags (labels may be missing or all -1)
        self.has_pseudo = pseudo_tr is not None and (pseudo_tr >= 0).any().item()
        self.has_true   = true_tr is not None and (true_tr >= 0).any().item()

        # If a label stream is missing, fill with -1 to maintain shapes
        self.batch_train  = _ensure_tensor(batch_tr)
        self.batch_test   = _ensure_tensor(batch_te)
        self.pseudo_train = _ensure_tensor(pseudo_tr if self.has_pseudo else torch.full_like(self.batch_train, -1))
        self.pseudo_test  = _ensure_tensor(pseudo_te if self.has_pseudo else torch.full_like(self.batch_test,  -1))
        self.true_train   = _ensure_tensor(true_tr if self.has_true   else torch.full_like(self.batch_train, -1))
        self.true_test    = _ensure_tensor(true_te if self.has_true   else torch.full_like(self.batch_test,  -1))

        # ----- build normalized second view using *batch* labels explicitly -----
        pre_Xtr_num, nb_stats = norm_batch(self.batch_train, Xtr_num,
                                           return_stats=True,
                                           prior_strength=100,
                                           min_cells_per_batch=10,
                                           robust_global=True,
                                           mix_global=0.1,
                                           clip_z=5.0)
        pre_Xte_num = norm_batch(self.batch_test, Xte_num, stats=nb_stats)

        # store numerical views
        self.X_train_num = Xtr_num.contiguous()
        self.pre_X_train_num = pre_Xtr_num.contiguous()
        self.X_test_num  = Xte_num.contiguous()
        self.pre_X_test_num = pre_Xte_num.contiguous()

        # ---------------- explicit-label dataset & loader ----------------
        self.train_data = SCDataset2View(
            self.X_train_num, self.pre_X_train_num,
            self.batch_train, self.pseudo_train, self.true_train
        )

        # Optional: batch-balanced sampler over *batch* labels
        if self.batch_train.numel() > 0:
            self.sampler = UniformLabelSampler(len(self.train_data), self.batch_train)
            shuffle_flag = False
        else:
            self.sampler = None
            shuffle_flag = True

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=shuffle_flag if self.sampler is None else False,
            num_workers=4,
            pin_memory=self.device_is_cuda,
            persistent_workers=(4 > 0)
        )

        # ---------------- model / opt ----------------        
        self.model = Model_VAE(
            self.num_layers, self.d_numerical, categories, self.d_token,
            n_head=self.n_head, factor=self.factor, bias=self.bias
        ).to(self.device)
        self.pre_encoder = Encoder_model(
            self.num_layers, self.d_numerical, categories, self.d_token,
            n_head=self.n_head, factor=self.factor, bias=self.bias
        ).to(self.device)
        self.pre_decoder = Decoder_model(
            self.num_layers, self.d_numerical, categories, self.d_token,
            n_head=self.n_head, factor=self.factor, bias=self.bias
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=WD)
        total_steps = self.num_epochs * len(self.train_loader) * 2
        self.training_steps = float(self.training_steps or total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_training_steps=self.training_steps,
            num_warmup_steps=100,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device_is_cuda)

        # meters
        self.losses = AverageMeter()
        self.train_acc = AverageMeter(); self.train_ari = AverageMeter()
        self.train_nmi = AverageMeter(); self.train_kbet = AverageMeter()
        self.best_nmi = 0.0

        # logs
        self.train_prob = pd.DataFrame()
        self.test_prob = pd.DataFrame()

        # prototypes created lazily
        self.proto_head: Optional[ProtoHead] = None

        # Debug summary so we *know* which labels exist and their sizes.
        if self.verbose:
            print(f"[Explicit labels] has_pseudo={self.has_pseudo}")
            print(f"[Explicit labels] batch_train/test: {self.batch_train.shape} / {self.batch_test.shape}")
            print(f"[Explicit labels] pseudo_train/test: {self.pseudo_train.shape} / {self.pseudo_test.shape}")            

        self.start_time = time.time()

    @staticmethod
    def _flatten_z(z: torch.Tensor) -> torch.Tensor:
        # keep your current choice (higher NMI typically)
        return z.view(z.size(0), -1) if z.dim() == 3 else z #z.mean(1)#
    

    def train(self):
        self._latents_cache = None
        epoch_bar = tqdm(total=self.num_epochs, desc="Training", leave=True)
        for epoch in range(self.num_epochs + 1):
            self.losses.reset(); self.train_acc.reset(); self.train_ari.reset(); self.train_nmi.reset(); self.train_kbet.reset()
            train_soft_prob = []

            sup_ramp = min(1.0, epoch / 5.0)
            center_ramp = min(1.0, epoch / 8.0)

            for batch_num, pre_batch_num, batch_lab, pseudo_lab, true_lab in self.train_loader:
                if self.device_is_cuda:
                    batch_num      = batch_num.to(self.device, non_blocking=True)
                    pre_batch_num  = pre_batch_num.to(self.device, non_blocking=True)
                    batch_lab      = batch_lab.to(self.device, non_blocking=True)
                    pseudo_lab     = pseudo_lab.to(self.device, non_blocking=True)
                    true_lab       = true_lab.to(self.device, non_blocking=True)

                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.device_is_cuda):
                    # ----- view 1 -----
                    z1, Recon_X_num1, Recon_X_cat1, mu_z1, std_z1 = self.model(batch_num, None)
                    loss_mse1, _, loss_kld1, _ = compute_loss(batch_num, None, Recon_X_num1, Recon_X_cat1, mu_z1, std_z1)
                    loss1 = loss_mse1 + 0.001 * loss_kld1
                    # reg1 = assign_soft_cluster(batch_lab, z1, self.num_clusters)

                    # ----- view 2 -----
                    z2, Recon_X_num2, Recon_X_cat2, mu_z2, std_z2 = self.model(pre_batch_num, None)
                    loss_mse2, _, loss_kld2, _ = compute_loss(pre_batch_num, None, Recon_X_num2, Recon_X_cat2, mu_z2, std_z2)
                    loss2 =  0.1 * loss_kld2        
                    # reg2 = assign_soft_cluster(batch_lab, z2, self.num_clusters)          

                    # ----- prototype clustering -----
                    z1f = self._flatten_z(z1); z2f = self._flatten_z(z2); zc = 0.5 * (z1f + z2f)
                    if self.proto_head is None:
                        self.proto_head = ProtoHead(d=zc.size(-1), K=self.num_clusters, tau=self.proto_tau,
                                                    cosine=True, ema_m=float(self.proto_ema_m)).to(self.device)
                        self.proto_head.init_from_batch(zc.detach())
                        self.optimizer.add_param_group({'params': self.proto_head.parameters(), 'lr': self.lr})

                    logits1 = self.proto_head.logits(z1f)
                    logits2 = self.proto_head.logits(z2f)
                    logits  = self.proto_head.logits(zc)

                    q1 = F.softmax(logits1, dim=1)
                    q2 = F.softmax(logits2, dim=1)
                    q  = F.softmax(logits,  dim=1)
                    p  = dec_targets(q)

                    L_clust  = F.kl_div(F.log_softmax(logits, dim=1), p, reduction='batchmean')
                    L_center = proto_center_loss(zc, self.proto_head, p) * center_ramp
                    L_repulse = prototype_repulsion(self.proto_head, target=float(self.repulse_tar))

                    with torch.no_grad():
                        conf, y_hat = q.max(dim=1)
                        mask = conf >= float(self.conf_thr)
                        y_use = y_hat[mask]
                    L_margin = logits.new_tensor(0.0)
                    if mask.any():
                        logits_m = cosface_margin_logits(logits[mask], y_use, m=float(self.cosface_m))
                        L_margin = F.cross_entropy(logits_m, y_use)

                    # ----- supervised proto loss (EXPLICITLY uses pseudo labels if available) -----
                    L_sup = logits.new_tensor(0.0)
                    if self.has_pseudo:
                        # Only use samples with valid pseudo label (>=0). Others are ignored.
                        valid = (pseudo_lab >= 0)
                        if valid.any():
                            L_sup = smooth_ce_loss(logits[valid], pseudo_lab[valid], eps=float(self.label_eps))

                    L_cons = sym_kl(q1, q2)

                    proto_loss = (
                        float(self.lam_clust)   * L_clust   +
                        float(self.lam_center)  * L_center  +
                        float(self.lam_repulse) * L_repulse +
                        (float(self.lam_margin) * sup_ramp) * L_margin +
                        (float(self.lam_sup_max) * sup_ramp) * L_sup +
                        float(self.lam_cons)    * L_cons
                    )

                    # ----- metrics (EXPLICITLY uses batch + true labels) -----
                    targets = true_lab if self.has_true else pseudo_lab
                    valid = pseudo_lab[pseudo_lab >= 0]
                    num_clusters = int(valid.unique().numel()) if valid.numel() else self.num_clusters

                    mask_ratio, acc, ari, nmi, kbet, soft_prob = calculate_metrics(
                        batch_labels=batch_lab,
                        true_labels=valid,         # None if not available
                        batch_z1=z1,
                        batch_z2=z2,
                        threshold=self.threshold,
                        num_clusters=num_clusters,
                        metrics=True
                    )

                    total_loss = loss1 + loss2 + proto_loss #+ 0.1*reg1 + 0.1*reg2

                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()

                with torch.no_grad():
                    self.proto_head.ema_update(zc.detach(), p.detach())

                # meters
                self.losses.update(float(total_loss))
                self.train_acc.update(acc); self.train_ari.update(ari)
                self.train_nmi.update(nmi); self.train_kbet.update(kbet)

                sp = soft_prob.detach().cpu().numpy()
                # Standardize to 2D (B, K)
                if sp.ndim == 0:              # rare scalar case
                    sp = sp.reshape(1, 1)
                elif sp.ndim == 1:            # (B,) -> (B, 1)
                    sp = sp.reshape(-1, 1)
                train_soft_prob.append(sp)

            # checkpoint on NMI
            if self.train_nmi.avg > self.best_nmi:
                self.best_nmi = self.train_nmi.avg
                torch.save(self.model.state_dict(), self.model_save_path)
                self.save_latents()

            # stack epoch probs
            train_soft_prob = np.concatenate(train_soft_prob, axis=0)
            K = int(train_soft_prob.shape[1])
            cols = [f"ep{epoch}_k{j}" for j in range(K)]
            df_epoch = pd.DataFrame(train_soft_prob, columns=cols)
            self.train_prob = pd.concat([self.train_prob, df_epoch], axis=1)

            # ---------------- eval (single pass on full test tensors) ----------------
            self.model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device_is_cuda):
                Xte  = self.X_test_num.to(self.device, non_blocking=True)
                preX = self.pre_X_test_num.to(self.device, non_blocking=True)
                z1, recon_test_num1, recon_test_cat1, mu1, std1 = self.model(Xte, None)
                z2, recon_test_pre_num1, recon_test_ore_cat1, mu2, std2 = self.model(preX, None)
                
                                
                test_mse1, _, kl1, _ = compute_loss(Xte,  None,  recon_test_num1, recon_test_cat1, mu1, std1)
                test_mse2, _, kl2, _ = compute_loss(preX, None, recon_test_pre_num1, recon_test_ore_cat1, mu2, std2)
                test_loss1 = test_mse1 + 0.001 * kl1
                test_loss2 = 0.10 * kl2

                test_proto_loss = torch.zeros(1, device=self.device, dtype=z1.dtype).squeeze()
                if self.proto_head is not None:
                    z1f = self._flatten_z(z1)
                    z2f = self._flatten_z(z2)
                    zc  = 0.5 * (z1f + z2f)

                    logits1 = self.proto_head.logits(z1f)
                    logits2 = self.proto_head.logits(z2f)
                    logits  = self.proto_head.logits(zc)

                    q1 = F.softmax(logits1, dim=1)
                    q2 = F.softmax(logits2, dim=1)
                    q  = F.softmax(logits,  dim=1)
                    p  = dec_targets(q)

                    L_clust   = F.kl_div(F.log_softmax(logits, dim=1), p, reduction='batchmean')
                    L_center  = proto_center_loss(zc, self.proto_head, p) * center_ramp
                    L_repulse = prototype_repulsion(self.proto_head, target=float(self.repulse_tar))

                    conf, y_hat = q.max(dim=1)
                    mask = conf >= float(self.conf_thr)
                    y_use = y_hat[mask]
                    L_margin = logits.new_tensor(0.0)
                    if mask.any():
                        logits_m = cosface_margin_logits(logits[mask], y_use, m=float(self.cosface_m))
                        L_margin = F.cross_entropy(logits_m, y_use)

                    L_sup = logits.new_tensor(0.0)
                    if self.has_pseudo:
                        pseudo_test = self.pseudo_test.to(self.device, non_blocking=True)
                        valid = (pseudo_test >= 0)
                        if valid.any():
                            L_sup = smooth_ce_loss(logits[valid], pseudo_test[valid], eps=float(self.label_eps))

                    L_cons = sym_kl(q1, q2)

                    test_proto_loss = (
                        float(self.lam_clust)   * L_clust   +
                        float(self.lam_center)  * L_center  +
                        float(self.lam_repulse) * L_repulse +
                        (float(self.lam_margin) * sup_ramp) * L_margin +
                        (float(self.lam_sup_max) * sup_ramp) * L_sup +
                        float(self.lam_cons)    * L_cons
                    )

                targets_test = self.true_test.to(self.device, non_blocking=True) if self.has_true else pseudo_test
                valid_test = pseudo_test[pseudo_test >= 0]
                batches_test = self.batch_test.to(self.device, non_blocking=True)

                test_mask_ratio, test_acc, test_ari, test_nmi, test_kbet, test_soft_prob = calculate_metrics(
                    batch_labels=batches_test,
                    true_labels=valid_test,
                    batch_z1=z1,
                    batch_z2=z2,
                    threshold=self.threshold,
                    num_clusters=num_clusters,
                    metrics=True
                )                
                
                test_total = test_loss1 + test_loss2 + test_proto_loss 

                test_sp = test_soft_prob.detach().cpu().numpy()
                if test_sp.ndim == 0:
                    test_sp = test_sp.reshape(1, 1)
                elif test_sp.ndim == 1:
                    test_sp = test_sp.reshape(-1, 1)

                K_test = int(test_sp.shape[1])
                test_cols = [f"test_ep{epoch}_k{j}" for j in range(K_test)]
                df_test = pd.DataFrame(test_sp, columns=test_cols)   
                self.test_prob = pd.concat([self.test_prob, df_test], axis=1)

            # wandb.log({
            #     'LR': self.optimizer.param_groups[0]['lr'],
            #     'Train loss_total': self.losses.avg,
            #     'Test loss_total': float(test_total),
            #     'Train Mask_ratio': mask_ratio,
            #     'Train ACC': self.train_acc.avg,
            #     'Train ARI': self.train_ari.avg,
            #     'Train NMI': self.train_nmi.avg,
            #     'Train kBET': self.train_kbet.avg,
            #     'Test Mask_ratio': test_mask_ratio,
            #     'Test ACC': test_acc,
            #     'Test ARI': test_ari,
            #     'Test NMI': test_nmi,
            #     'Test kBET': test_kbet,
            #     'proto_tau': self.proto_tau,
            #     'lam_clust': self.lam_clust,
            #     'lam_center': self.lam_center,
            #     'lam_sup_max': self.lam_sup_max,
            #     'lam_margin': self.lam_margin,
            #     'cosface_m': self.cosface_m,
            #     'lam_repulse': self.lam_repulse,
            #     'repulse_target': self.repulse_tar,
            #     'lam_cons': self.lam_cons,
            #     'conf_thr': self.conf_thr,                
            #     'label_eps': self.label_eps
            # })

            if self.verbose:
                print({k: round(float(v), 3) for k, v in {
                    'Train NMI': self.train_nmi.avg, 'Train ARI': self.train_ari.avg,
                    'Train ACC': self.train_acc.avg, 'Train kBET': self.train_kbet.avg,
                    'Train CCR': mask_ratio
                }.items()})
                # print(f'Train total loss: {self.losses.avg:.6f}   Test total loss: {float(test_total):.6f}')
            epoch_bar.set_postfix({"epoch": f"{epoch+1}/{self.num_epochs}",
                                   "Train_loss": f"{self.losses.avg:.4f}",
                                   "Test_loss": f"{float(test_total):.4f}"})
            epoch_bar.update(1)

        mins = (time.time() - self.start_time)/60
        if self.verbose:
            print(f'Training time: {mins:.2f} mins')

        # save soft prob curves
        # outdir = f'{self.curr_dir}/plots/fig2_eval_training'
        # os.makedirs(outdir, exist_ok=True)
        # self.train_prob.to_csv(f'{outdir}/train_prob.csv', index=False)
        # self.test_prob.to_csv(f'{outdir}/test_prob.csv', index=False)
        # return test_nmi

    @torch.no_grad()
    def _collect_latents(self) -> torch.Tensor:        
        if os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        self.pre_encoder.load_weights(self.model)        

        # concat on CPU, then move to device for encoding
        X_all     = torch.cat([self.X_train_num,     self.X_test_num],     dim=0).to(self.device, non_blocking=True)
        pre_X_all = torch.cat([self.pre_X_train_num, self.pre_X_test_num], dim=0).to(self.device, non_blocking=True)      
        batch_all = torch.cat([self.batch_train, self.batch_test], dim=0).to(self.device, non_blocking=True)

        z_all = split_batch_precoder(X_all, pre_X_all, None, batch_all, self.pre_encoder, self.batch_size, self.prior_pcr)

        # reorder to original AnnData order if we know the split indices
        z_out = z_all
        if getattr(self, "split_idx", None) is not None:
            tr_idx, te_idx = self.split_idx            
            z_reordered = torch.empty(z_all.size(), dtype=z_all.dtype, device=z_all.device)
            z_reordered[torch.as_tensor(tr_idx, device=z_all.device)] = z_all[:len(tr_idx)]
            z_reordered[torch.as_tensor(te_idx, device=z_all.device)] = z_all[len(tr_idx):]
            z_out = z_reordered

        self._latents_cache = z_out.detach().cpu()
        return self._latents_cache

    @torch.no_grad()
    def save_latents(self):
        latents = self._collect_latents()
        torch.save(latents, f"{self.ckpt_dir}/all_latents.pt")
        torch.save(self.pre_encoder.state_dict(), self.encoder_save_path)
        if self.verbose:
            print(f"Saved whole dataset latents (N={latents.shape[0]})")

    def get_latent_representation(
        self,        
        n_components: Optional[int] = 50,        
        **pca_kwargs,
    ):
        """
        Return PCA-compressed latent vectors suitable for ``adata.obsm``.

        Parameters
        ----------
        n_components:
            Number of principal components to retain. When ``None`` the raw
            flattened latents are returned.
        **pca_kwargs:
            Additional keyword arguments forwarded to ``sklearn.decomposition.PCA``.
        """
        alias = pca_kwargs.pop("n_compoents", None)
        if alias is not None:
            n_components = alias

        latents = self._collect_latents()
        flat = latents.view(latents.size(0), -1).detach().cpu().numpy()

        if n_components is None:
            return flat

        from sklearn.decomposition import PCA

        pca_kwargs.setdefault("svd_solver", "arpack")
        pca_kwargs.setdefault("random_state", 42)
        n_components = int(min(max(1, n_components), flat.shape[1]))
        pca = PCA(n_components=n_components, **pca_kwargs)
        return pca.fit_transform(flat)



# --------------------------------- main ---------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Autoencoder')    
    parser.add_argument('--adata', type=str, required=True, help='Path to an .h5ad file')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--num_clusters', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--training_steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', help='Log detailed training information each epoch')
    args = parser.parse_args()

    args.device = f'cuda:{args.gpu}' if (args.gpu != -1 and torch.cuda.is_available()) else 'cpu'    

    import scanpy as sc
    adata = sc.read_h5ad(args.adata)

    model = VAEModel(
        adata,
        var_key='scBIOT_OT',
        batch_key='batch',
        pseudo_key='leiden_scBIOT_OT',
        true_key='cell_type',        
        num_clusters=args.num_clusters,
        random_seed=args.random_seed,
        training_steps=args.training_steps,
        batch_size = args.batch_size,
        lr = args.lr,
        num_epochs = args.num_epochs,
        device=args.device,
        hyperparams=HyperParams(),
        verbose=args.verbose,
    )
    model.train()
    adata.obsm['scBIOT'] = model.get_latent_representation(n_components=50)

    
