import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import time
import random
import os
from torch.utils.data.sampler import Sampler


BATCH_SIZE = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class SCDataset(Dataset):
    def __init__(self, X_barcode, X_num, pre_X_num, X_cat):
        self.X_barcode = X_barcode
        self.X_num = X_num
        self.pre_X_num = pre_X_num
        self.X_cat = X_cat

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        dataset_barcode = self.X_barcode[idx]
        dataset_num = self.X_num[idx]
        pre_dataset_num = self.pre_X_num[idx] if self.pre_X_num is not None and len(self.pre_X_num)> 0 else  torch.empty(0)
        dataset_cat = self.X_cat[idx] if self.X_cat is not None and len(self.X_cat) > 0 else  torch.empty(0)

        if self.pre_X_num is not None and len(self.pre_X_num)> 0:
            sample = (dataset_barcode, dataset_num, pre_dataset_num, dataset_cat) if len(self.X_cat) > 0 else (dataset_barcode, dataset_num, pre_dataset_num, [])
        else:
            sample = (dataset_barcode, dataset_num, [], dataset_cat) if len(self.X_cat) > 0 else (dataset_barcode, dataset_num, [], [])

        return sample




def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=1.0, #7. / 16.,
                                    num_warmup_steps=50,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)




# def norm_batch(X_train_cat, X_train_num):
#     unique_classes = torch.unique(X_train_cat)

#     # Initialize normalized tensor with the same shape as X_train_num
#     X_train_num_normalized = X_train_num.clone()

#     # Normalize X_train_num based on the classes in X_train_cat[:, 0]
#     for cls in unique_classes:
#         cls_mask = (X_train_cat == cls).squeeze()
#         cls_values = X_train_num[cls_mask]

#         cls_mean = cls_values.mean(dim=0)
#         cls_std = cls_values.std(dim=0)

#         # Avoid division by zero
#         cls_std[cls_std == 0] = 1.0

#         X_train_num_normalized[cls_mask] = (cls_values - cls_mean) / cls_std

#     return X_train_num_normalized

@torch.no_grad()
def norm_batch(
    x_cat: torch.Tensor,
    x_num: torch.Tensor,
    stats: dict | None = None,
    return_stats: bool = False,
    prior_strength: int = 200,     # larger -> more shrink toward global for small batches
    min_cells_per_batch: int = 64, # batches with < this many cells use global stats
    robust_global: bool = True,    # use median/MAD for global stats
    mix_global: float = 0.35,      # convex mix of per-batch and global stats (0..1)
    clip_z: float = 5.0,           # clamp extreme z-scores to stabilize rare cells
    eps: float = 1e-6
):
    """
    Isolation-preserving batch normalization for view-2.

    Returns:
      - if return_stats=False: normalized tensor
      - if return_stats=True: (normalized tensor, stats dict)
    """
    x_cat = x_cat.view(-1)
    device = x_num.device

    # ---------- FIT STATS (on TRAIN ONLY) ----------
    if stats is None:
        # Global robust stats
        if robust_global:
            g_mu = x_num.median(dim=0).values
            g_mad = (x_num - g_mu).abs().median(dim=0).values + eps
            g_std = 1.4826 * g_mad
        else:
            g_mu = x_num.mean(dim=0)
            g_std = x_num.std(dim=0, unbiased=False) + eps

        stats = {
            "global_mu": g_mu.to(device),
            "global_std": g_std.to(device),
            "per_batch": {},      # int(batch_id) -> (mu_hat, std_hat, n)
            "prior_strength": prior_strength,
            "min_cells_per_batch": min_cells_per_batch,
            "mix_global": mix_global,
            "robust_global": robust_global,
            "clip_z": clip_z,
        }

        unique_batches = torch.unique(x_cat)
        for b in unique_batches:
            mask = (x_cat == b)
            n = int(mask.sum().item())
            xb = x_num[mask]
            # raw batch stats
            b_mu = xb.mean(dim=0)
            b_std = xb.std(dim=0, unbiased=False) + eps
            # shrink toward global (weight depends on n)
            a = prior_strength / (prior_strength + max(n, 1))  # 0..1
            mu_hat  = (1.0 - a) * b_mu + a * g_mu
            std_hat = torch.sqrt((1.0 - a) * (b_std ** 2) + a * (g_std ** 2))
            # mix some global to avoid over-correction (helps isolated labels)
            if mix_global > 0:
                mu_hat  = (1.0 - mix_global) * mu_hat  + mix_global * g_mu
                std_hat = (1.0 - mix_global) * std_hat + mix_global * g_std
            stats["per_batch"][int(b.item())] = (mu_hat.to(device), std_hat.to(device), n)

    # ---------- APPLY STATS ----------
    g_mu, g_std = stats["global_mu"], stats["global_std"]
    out = torch.empty_like(x_num)
    clip_z = stats.get("clip_z", clip_z)
    min_cells = stats.get("min_cells_per_batch", min_cells_per_batch)

    for b in torch.unique(x_cat):
        bi = int(b.item())
        mask = (x_cat == b)
        # fallback to global if unseen batch or too small
        if (bi not in stats["per_batch"]) or (stats["per_batch"][bi][2] < min_cells):
            mu_hat, std_hat = g_mu, g_std
        else:
            mu_hat, std_hat, _ = stats["per_batch"][bi]
        z = (x_num[mask] - mu_hat) / std_hat
        if clip_z is not None and clip_z > 0:
            z = z.clamp_(-clip_z, clip_z)
        out[mask] = z

    return (out, stats) if return_stats else out




class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class UniformLabelSampler(Sampler[int]):
    """
    Samples elements uniformly across labels (pseudo labels).
    Accepts torch.Tensor or numpy array; regenerates a fresh, balanced
    index list every epoch.
    """
    def __init__(self, N: int, labels, generator: torch.Generator | None = None):
        self.N = int(N)
        # Coerce labels -> 1D CPU LongTensor
        if isinstance(labels, np.ndarray):
            labels = torch.as_tensor(labels, dtype=torch.long)
        elif not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long)
        self.labels = labels.view(-1).long().cpu()

        self.generator = generator  # optional torch.Generator for reproducibility
        self._build_groups()

    def _build_groups(self):
        # unique labels and inverse index map
        unique, inv = torch.unique(self.labels, return_inverse=True, sorted=True)
        self.groups = [torch.where(inv == i)[0] for i in range(len(unique))]
        self.num_groups = len(self.groups)

    def _sample_once(self) -> list[int]:
        if self.num_groups == 0:
            return list(range(self.N))  # degenerate case

        size_per = max(1, math.ceil(self.N / self.num_groups))
        chunks = []
        g = self.generator
        for idxs in self.groups:
            if idxs.numel() == 0:
                continue
            # sample with replacement to ensure enough examples from small groups
            sel = idxs[torch.randint(low=0, high=idxs.numel(), size=(size_per,), generator=g)]
            chunks.append(sel)

        out = torch.cat(chunks) if len(chunks) else torch.empty(0, dtype=torch.long)
        # shuffle
        perm = torch.randperm(out.numel(), generator=g) if out.numel() else out
        out = out[perm] if out.numel() else out

        # trim/pad to exactly N
        if out.numel() < self.N and out.numel() > 0:
            out = torch.cat([out, out[: self.N - out.numel()]])
        else:
            out = out[: self.N]
        return out.tolist()

    def __iter__(self):
        # regenerate a fresh balanced ordering each epoch
        return iter(self._sample_once())

    def __len__(self) -> int:
        return self.N



@torch.no_grad()
def kbet_evaluate(all_z, batches):
    # kbet
    # https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html

    import scib_metrics
    from sklearn.neighbors import NearestNeighbors
    
    all_z = all_z
    batches = batches

    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(all_z)
    distances, indices = nbrs.kneighbors(all_z)

    nbrs_res = scib_metrics.nearest_neighbors.NeighborsResults(indices, distances)
    kbet = scib_metrics.kbet(nbrs_res, batches, alpha=0.05)
    kbet = kbet[0]

    # _, preds_top5 = probs.topk(5, 1, largest=True)
    # reordered_preds_top5 = torch.zeros_like(preds_top5)
    # for pred_i, target_i in match:
    #     reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    # correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
    # top5 = float(correct_top5_binary.sum()) / float(num_elems)
    return kbet
    


@torch.no_grad()
def hungarian_evaluate(all_predictions,
                       class_names=None,
                       compute_purity=True,
                       compute_confusion_matrix=False,
                       confusion_matrix_file=None):
    """
    Robust to NumPy/Torch inputs; runs fully on CPU; no .cuda() calls.
    """

    # --------- helpers ---------
    def _to_long_tensor(x):
        if torch.is_tensor(x):
            return x.detach().to(dtype=torch.long, device='cpu').view(-1)
        return torch.as_tensor(np.asarray(x), dtype=torch.long, device='cpu').view(-1)

    def _to_float2d(x):
        if torch.is_tensor(x):
            x = x.detach().to(dtype=torch.float32, device='cpu')
        else:
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device='cpu')
        return x

    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # --------- unpack & normalize dtypes ---------
    head        = all_predictions
    z_t         = _to_float2d(head['all_z'])
    batches_t   = _to_long_tensor(head['batches'])
    targets_t   = _to_long_tensor(head['targets'])
    preds_t     = _to_long_tensor(head['predictions'])
    probs_t     = _to_float2d(head.get('probabilities', np.zeros((len(preds_t), 0), dtype=np.float32)))

    num_classes = int(torch.unique(targets_t).numel())
    num_elems   = int(targets_t.numel())

    # --------- hungarian matching (expects Torch tensors) ---------
    match = _hungarian_match(preds_t, targets_t, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.empty_like(preds_t)
    for pred_i, target_i in match:
        mask = (preds_t == int(pred_i))
        if mask.any():
            reordered_preds[mask] = int(target_i)

    # --------- metrics ---------
    acc = float((reordered_preds == targets_t).float().mean().item())

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix as _cm
    nmi = float(normalized_mutual_info_score(_to_numpy(targets_t), _to_numpy(preds_t)))
    ari = float(adjusted_rand_score(_to_numpy(targets_t), _to_numpy(preds_t)))

    # kBET: pass numpy, and tolerate tuple returns
    kbet_out = kbet_evaluate(_to_numpy(z_t), _to_numpy(batches_t))
    if isinstance(kbet_out, (tuple, list, np.ndarray)):
        kbet = float(kbet_out[0])
    else:
        kbet = float(kbet_out)

    if compute_confusion_matrix:
        cm = _cm(_to_numpy(reordered_preds), _to_numpy(targets_t))
        if confusion_matrix_file:
            np.savetxt(confusion_matrix_file, cm, fmt='%d', delimiter=',')

    return acc, ari, nmi, kbet


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    from scipy.optimize import linear_sum_assignment
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res



from typing import Optional
import numpy as np
import torch

def calculate_metrics(
    batch_labels: Optional[torch.Tensor],
    true_labels: Optional[torch.Tensor],
    batch_z1: torch.Tensor,
    batch_z2: torch.Tensor,
    threshold: float = 0.1,
    num_clusters: int = 30,
    metrics: bool = True,
    use_true_labels: bool = True,  # kept for interface compatibility
):
    """
    Compute clustering + batch-mixing metrics from two views (z1, z2).

    Returns:
        mask_ratio (% of confident cells),
        acc, ari, nmi,
        kbet,
        soft_prob (tensor on the same device as inputs)

    Notes:
        - If true_labels is None, (acc, ari, nmi) are returned as NaN.
        - If batch_labels is None, kBET is returned as NaN.
    """
    with torch.no_grad():
        # ---- per-cell embedding (flatten tokens if needed) ----
        def _flat(z: torch.Tensor) -> torch.Tensor:
            return z.view(z.size(0), -1)

        z = 0.5 * (_flat(batch_z1) + _flat(batch_z2))          # [B, D]
        B = z.size(0)
        K = int(num_clusters)

        # ---- k-means on z (returns hard clusters + centers) ----
        cl, centers = kmeans(z, K=K, Niter=10)                 # centers: [K, D]

        # ---- DEC-style soft assignments from distances ----
        dist2 = torch.cdist(z, centers, p=2.0).pow(2)          # [B, K]
        alpha = 1.0
        q = (1.0 / (1.0 + dist2 / alpha)).pow((alpha + 1.0) / 2.0)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
        soft_prob, soft_cluster = q.max(dim=1)

        # ---- confident subset ----
        mask = soft_prob > threshold
        mask_ratio = float(mask.sum().item()) * 100.0 / float(B)

        # ---- metrics (defaults) ----
        acc = ari = nmi = float('nan')
        kbet = float('nan')

        # Prepare numpy copies if available
        batches_np = None
        if batch_labels is not None:
            batches_np = batch_labels.detach().cpu().numpy()

        # Supervised clustering metrics (require true_labels)
        if metrics and (true_labels is not None):
            eval_data = {
                'batches': batches_np,
                'all_z': z.detach().cpu().numpy(),
                'targets': true_labels.detach().cpu().numpy(),
                'predictions': soft_cluster.detach().cpu().numpy(),
                'probabilities': soft_prob.detach().cpu().numpy(),
            }
            acc, ari, nmi, kbet = hungarian_evaluate(eval_data)

        # Batch-mixing only path (when not computing supervised metrics)
        elif not metrics:
            if batches_np is not None:
                kbet = kbet_evaluate(z.detach().cpu().numpy(), batches_np)
            else:
                kbet = float('nan')

        return mask_ratio, acc, ari, nmi, kbet, soft_prob




def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



def kmeans(x, K=10, Niter=10, verbose=False):
    """Implements Lloyd's algorithm for the Euclidean metric with K-means++ initialization."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means++ initialization
    c = torch.empty((K, D), dtype=x.dtype, device=x.device)
    c[0] = x[torch.randint(0, N, (1,))]  # Randomly choose the first centroid

    for k in range(1, K):
        D_ij = torch.cdist(x, c[:k], p=2)  # Calculate distances to the already chosen centroids
        D2 = D_ij.min(dim=1)[0]  # Choose the minimum distance to any centroid
        prob = D2 / D2.sum()  # Probability proportional to distance squared
        next_idx = torch.multinomial(prob, 1)  # Sample the next centroid
        c[k] = x[next_idx]

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = torch.cdist(x, c, p=2)  # (N, K) pairwise distances
        cl = D_ij.argmin(dim=1).long()  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        for k in range(K):
            if torch.any(cl == k):
                c[k] = x[cl == k].sum(dim=0)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        Ncl = Ncl + 1e-10  # Avoid division by zero
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c




# def poe_shared(mu1, std1, mu2, std2, prior_var=1.0, eps=1e-6):
#     # Do this in fp32 for stability even under autocast
#     mu1 = mu1.float(); mu2 = mu2.float()
#     var1 = (std1.float()**2).clamp(min=eps, max=1e6)
#     var2 = (std2.float()**2).clamp(min=eps, max=1e6)

#     prec1 = 1.0 / var1
#     prec2 = 1.0 / var2
#     prec0 = 1.0 / float(prior_var)

#     prec = prec1 + prec2 + prec0
#     mu   = (mu1*prec1 + mu2*prec2) / prec
#     var  = 1.0 / prec

#     logvar = var.clamp(min=eps).log()
#     return mu, logvar  





# def split_batch_precoder(X_train_num, pre_X_train_num, X_train_cat, batch_labels, pre_encoder, BATCH_SIZE):
#     num_samples = X_train_num.shape[0]
#     results1, results2 = [], []
#     mu_vec1, mu_vec2 = [], []
#     std_vec1, std_vec2 = [], []
#     batch_labels = batch_labels

#     for i in range(0, num_samples, BATCH_SIZE):
#         end_idx = min(i + BATCH_SIZE, num_samples)

#         batch_num = X_train_num[i:end_idx]
#         batch_cat = X_train_cat[i:end_idx] if X_train_cat is not None else None
#         pre_batch_num = pre_X_train_num[i:end_idx] if pre_X_train_num is not None else torch.empty(0, device=X_train_num.device)

#         z1, mu1, std1 = pre_encoder(batch_num, batch_cat ,return_all=True)
#         z2, mu2, std2 = pre_encoder(pre_batch_num, batch_cat, return_all=True) if pre_X_train_num is not None else torch.empty(0, device=X_train_num.device)

#         results1.append(z1)
#         results2.append(z2)
#         mu_vec1.append(mu1)
#         mu_vec2.append(mu2)
#         std_vec1.append(std1)
#         std_vec2.append(std2)

#     train_z1 = torch.cat(results1, dim=0)
#     train_z2 = torch.cat(results2, dim=0)
#     train_mu1 = torch.cat(mu_vec1, dim=0)
#     train_mu2 = torch.cat(mu_vec2, dim=0)
#     train_std1 = torch.cat(std_vec1, dim=0)
#     train_std2 = torch.cat(std_vec2, dim=0)
    
#     mu_sh, logvar_sh = poe_shared(train_mu1, train_std1, train_mu2, train_std2)
#     z_shared = mu_sh  # deterministic readout for clustering/metrics   

#     return z_shared#train_z1 + train_z2 if len(train_z2) > 0 else train_z1 # torch.cat([train_z1, train_z2], dim=1)#train_z1 + train_z2 if len(train_z2) > 0 else train_z1



# =================== paste-and-run: poe_shared_pcr_no_pseudo.py ===================
import torch

@torch.no_grad()
def _eta2_per_dim(mu: torch.Tensor, groups: torch.Tensor, eps: float = 1e-8):
    """
    Correlation ratio η² per feature: fraction of variance explained by 'groups'.

    Accepts 2D or 3D mu:
      - [N, D] or
      - [N, T, D] (e.g., token axis kept; returns [T, D])
    groups: [N] integer labels
    returns: per-feature tensor in [0,1] with shape mu.shape[1:]
    """
    device = mu.device
    groups = groups.view(-1)
    mu_all = mu.mean(dim=0)
    ss_tot = ((mu - mu_all)**2).sum(dim=0).clamp_min(eps)
    ss_between = torch.zeros_like(ss_tot, device=device)
    for g in torch.unique(groups):
        m = (groups == g)
        if not torch.any(m):
            continue
        n_g = float(m.sum().item())
        mu_g = mu[m].mean(dim=0)
        ss_between += n_g * (mu_g - mu_all)**2
    return (ss_between / ss_tot).clamp(0.0, 1.0)


@torch.no_grad()
def poe_shared_pcr(
    mu1, std1, mu2, std2,
    batch_ids: torch.Tensor,
    y_pseudo: torch.Tensor | None = None,   # kept for compatibility; IGNORED
    *,
    prior_pcr: float = 1.0,           # base prior precision (1/var); ↑ -> PCR↑
    k_var: float = 2.0,                     # variance inflation scale for batchy dims
    k_prior: float = 3.0,                   # how strongly prior shrinks batchy dims
    bio_gamma: float = 0.6,                 # kept for compatibility; IGNORED
    eps: float = 1e-6
):
    """
    PCR-tilted Product-of-Experts over two encoders' Gaussians (NO pseudo labels).
    In batch-dominated dimensions, increase variance (downweight expert) and add
    prior precision to shrink toward 0.

    Shapes:
      mu*, std*: [N, D] or [N, T, D] (std* is stdev, not logvar)
      batch_ids: [N]
    Returns:
      mu: [N, D] or [N, T, D]
      logvar: same shape as mu
    """
    # Work in fp32 for stability
    mu1 = mu1.float(); mu2 = mu2.float()
    var1 = (std1.float()**2).clamp(min=eps, max=1e6)
    var2 = (std2.float()**2).clamp(min=eps, max=1e6)

    # Per-dimension batch dominance for each expert
    eta_b1 = _eta2_per_dim(mu1, batch_ids)        # [D] or [T, D]
    eta_b2 = _eta2_per_dim(mu2, batch_ids)        # [D] or [T, D]
    eta_b  = 0.5 * (eta_b1 + eta_b2)              # shared view of batchiness

    # ---- Tilt the experts' variances in batchy dims (downweight them) ----
    var1_adj = var1 * (1.0 + k_var * eta_b1).clamp_min(eps)
    var2_adj = var2 * (1.0 + k_var * eta_b2).clamp_min(eps)

    prec1 = 1.0 / var1_adj
    prec2 = 1.0 / var2_adj

    # ---- Feature-wise prior precision: shrink batchy dims toward 0 ----
    # No biology protection term (no pseudo labels).
    prior_prec = prior_pcr * (1.0 + k_prior * eta_b)
    prior_prec = prior_prec.clamp(min=0.0)

    # ---- PoE combine ----
    prec = prec1 + prec2 + prior_prec
    mu   = (mu1 * prec1 + mu2 * prec2) / prec
    var  = 1.0 / prec
    logvar = var.clamp(min=eps).log()
    return mu, logvar


@torch.no_grad()
def split_batch_precoder(
    X_train_num, pre_X_train_num, X_train_cat, batch_labels, pre_encoder, BATCH_SIZE, prior_pcr=1.0):
    """
    Returns z_shared (mean of PoE), with PCR-tilted fusion.
    Assumes pre_encoder(x_num, x_cat, return_all=True) -> (z, mu, std).
    Pseudo labels are NOT used.
    """
    N = X_train_num.shape[0]
    mu_vec1, mu_vec2, std_vec1, std_vec2 = [], [], [], []

    for i in range(0, N, BATCH_SIZE):
        j = min(i + BATCH_SIZE, N)
        x1 = X_train_num[i:j]
        xcat = X_train_cat[i:j] if X_train_cat is not None else None
        x2 = pre_X_train_num[i:j] if pre_X_train_num is not None else None

        z1, mu1, std1 = pre_encoder(x1, xcat, return_all=True)
        # drop special/CLS token if present on dim=1
        if mu1.dim() == 3 and mu1.size(1) > 1:
            mu1, std1 = mu1[:, 1:, :], std1[:, 1:, :]

        if x2 is not None:
            z2, mu2, std2 = pre_encoder(x2, xcat, return_all=True)
            if mu2.dim() == 3 and mu2.size(1) > 1:
                mu2, std2 = mu2[:, 1:, :], std2[:, 1:, :]
        else:
            # Fallback: use the same view twice if only one view is present
            z2, mu2, std2 = z1, mu1, std1

        mu_vec1.append(mu1); mu_vec2.append(mu2)
        std_vec1.append(std1); std_vec2.append(std2)

    train_mu1  = torch.cat(mu_vec1,  dim=0)
    train_mu2  = torch.cat(mu_vec2,  dim=0)
    train_std1 = torch.cat(std_vec1, dim=0)
    train_std2 = torch.cat(std_vec2, dim=0)

    mu_sh, logvar_sh = poe_shared_pcr(
        train_mu1, train_std1, train_mu2, train_std2,
        batch_ids=batch_labels.view(-1).to(train_mu1.device),
        y_pseudo=None,                    # explicitly ignored
        prior_pcr=prior_pcr,              # tune: 1.0–2.0
        k_var=2.0,                        # tune: 1.5–3.0  (↑ PCR with higher)
        k_prior=3.0                       # tune: 2.0–6.0  (↑ PCR with higher)
    )

    # Deterministic readout for clustering/metrics
    z_shared = mu_sh
    return  z_shared # 0.5*(train_mu1 + train_mu2) #



def split_batch_decoder(syn_data, model, BATCH_SIZE):
    # batch processing to reduce memory usage
    results = []
    num_results, cat_results = [], []
    num_samples = syn_data.shape[0]
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_samples)

        batch_num = syn_data[start_idx:end_idx]
        train_z_num, train_z_cat = model(batch_num)
        num_results.append(train_z_num)
        cat_results.append(train_z_cat)

    train_z_nums = torch.cat(num_results, dim=0)
    train_z_cats = [[torch.cat([cat_results[i][j] for i in range(len(cat_results))]) for j in range(len(cat_results[0]))]]

    return train_z_nums, train_z_cats[0]


def assign_soft_cluster(X_cat, train_test_z, num_clusters=30, alpha=1.0, info=2., entropy=0.4, eps=1e-8):
   
    # kmeans
    train_test_z = train_test_z.view(train_test_z.size(0), -1)
    cl, cluster_centers = kmeans(train_test_z, K=num_clusters, Niter=10)

    # probability of each cell to each cluster
    q = 1.0 / (1.0 + torch.sum((train_test_z.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha + eps)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / (torch.sum(q, dim=1) + eps)).t()

    # soft distribution of clusters
    # X_cat = X_cat[:, 0]
    unique_label_length = len(torch.unique(X_cat))

    # conditional cluster entropy
    cond_cluster_entropy = -torch.sum(q * torch.log(q + eps)) / q.size(0)
    # cluster entropy
    cluster_entropy = -torch.sum(torch.mean(q, dim=0) * torch.log(torch.mean(q, dim=0) + eps))

    # fair loss
    O = torch.zeros((num_clusters, unique_label_length), device=q.device)
    for b in range(unique_label_length):
        O[:, b] = torch.sum(q[X_cat == b], dim=0)

    # Normalize to get the joint probability
    pcg = O / (torch.sum(O) + eps)

    # Calculate marginal probabilities
    pc = torch.sum(pcg, dim=1, keepdim=True)
    pg = torch.sum(pcg, dim=0, keepdim=True)

    # Calculate mutual information
    info_fair_loss = torch.sum(pcg * torch.log(pcg / (pc * pg) + eps))

    reg_loss = info * info_fair_loss + entropy * (-cluster_entropy + cond_cluster_entropy)

    return reg_loss