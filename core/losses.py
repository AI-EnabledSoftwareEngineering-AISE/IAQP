
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

def l2n_np(x: np.ndarray, eps=1e-8) -> np.ndarray:
    """L2 normalize numpy array."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def identity_loss_barycentric(q_proj: torch.Tensor, t_orig: torch.Tensor,
                             pos_img_ids: torch.Tensor, X_base: torch.Tensor, 
                             beta: float = 0.15) -> torch.Tensor:
    """
    Barycentric semantic anchor that moves identity target toward relevant images.
    
    Args:
        q_proj: [B,D] projected query vectors (L2-normalized)
        t_orig: [B,D] original text vectors (L2-normalized)
        pos_img_ids: [B,K] positive image IDs per query
        X_base: [N,D] base image vectors (L2-normalized)
        beta: Mixing factor (0.1-0.2 recommended)
    
    Returns:
        scalar loss: distance from projected query to semantic center
    """
    B, K = pos_img_ids.shape
    
    losses = []
    for b in range(B):
        # Get positive images for this query
        pos_ids = pos_img_ids[b]  # [K]
        pos_images = X_base[pos_ids]  # [K,D]
        
        # Compute rank-weighted barycenter
        weights = torch.exp(-0.1 * torch.arange(K, device=pos_images.device, dtype=pos_images.dtype))
        weights = weights / weights.sum()
        
        # Weighted average of positive images
        pos_barycenter = (pos_images * weights.unsqueeze(1)).sum(dim=0)  # [D]
        pos_barycenter = F.normalize(pos_barycenter, dim=0)  # L2-normalize
        
        # Semantic center: mix original text with positive barycenter
        semantic_center = (1 - beta) * t_orig[b] + beta * pos_barycenter
        semantic_center = F.normalize(semantic_center, dim=0)
        
        # Distance from projected query to semantic center
        loss = ((q_proj[b] - semantic_center) ** 2).sum()
        losses.append(loss)
    
    return torch.stack(losses).mean()


def identity_loss_legacy(q_proj: torch.Tensor, t_orig: torch.Tensor) -> torch.Tensor:
    """
    Legacy L2-based identity loss for comparison.
    
    Args:
        q_proj: [B,D] projected query vectors (L2-normalized)
        t_orig: [B,D] original text vectors (L2-normalized)
    
    Returns:
        scalar loss: L2 distance between projected and original
    """
    return (q_proj - t_orig).pow(2).sum(dim=1).mean()


# -------------------- Smart Identity Loss --------------------
def identity_loss_cone(q_proj: torch.Tensor, t_orig: torch.Tensor, budget: int = None, epoch: int = 1) -> torch.Tensor:
    """
    Cone-based smart identity loss that allows helpful moves while preventing semantic drift.
    
    Args:
        q_proj: [B,D] projected query vectors (L2-normalized)
        t_orig: [B,D] original text vectors (L2-normalized)
        budget: Current training budget (for adaptive threshold)
        epoch: Current epoch (for decay)
    
    Returns:
        scalar loss: one-sided cosine barrier that fires only if rotation is too far
    """
    # Compute cosine similarity
    cosine_sim = (q_proj * t_orig).sum(dim=1)  # [B]
    
    # Adaptive threshold based on budget and epoch
    if budget is not None:
        # More permissive for small budgets (need more steering)
        budget_factor = max(0.85, 0.95 - 0.1 * (budget - 10) / 90.0)
    else:
        budget_factor = 0.9
    
    # Decay threshold across epochs (start permissive, gradually tighten)
    epoch_factor = max(0.8, 0.95 - 0.15 * (epoch - 1) / 10.0)
    
    # Use the more restrictive of budget and epoch factors
    cos_threshold = min(budget_factor, epoch_factor)
    
    # One-sided barrier: only penalize if cosine similarity drops below threshold
    cone_violation = F.relu(cos_threshold - cosine_sim)
    
    return cone_violation.mean()


# -------------------- Losses --------------------
def listwise_kld(q_vecs: torch.Tensor,
                 Xc: torch.Tensor,
                 mask: torch.Tensor,
                 teacher_P: torch.Tensor,
                 tau: float) -> torch.Tensor:
    s = torch.einsum('bd,bcd->bc', q_vecs, Xc).masked_fill(~mask, -1e9)
    student = torch.log_softmax(s / tau, dim=1)
    P = torch.where(mask, teacher_P, torch.zeros_like(teacher_P))
    P = P / (P.sum(dim=1, keepdim=True) + 1e-12)
    return F.kl_div(student, P, reduction='batchmean') * (tau ** 2)


def frontier_gap_loss(scores: torch.Tensor,
                      mask: torch.Tensor,
                      k: int,
                      m: float = 0.03,
                      extra: int = 32) -> torch.Tensor:
    scores = scores.masked_fill(~mask, -1e9)
    valid_counts = mask.sum(dim=1)
    eligible = valid_counts >= (k + 1)
    if not torch.any(eligible):
        return scores.new_zeros(())

    s_elig = scores[eligible]
    vc_min = int(valid_counts[eligible].min().item())
    Cwant = min(s_elig.size(1), max(k + 1, min(k + extra, vc_min)))
    top_vals, _ = torch.topk(s_elig, k=Cwant, dim=1, largest=True)

    kth = top_vals[:, k - 1]
    k1th = top_vals[:, k]
    gap = kth - k1th
    return F.relu(m - gap).mean()






def cell_ce_loss(q: torch.Tensor, 
                 y_cells: torch.Tensor, 
                 coarse_head: 'CoarseCellHead', 
                 tau: float = 0.07) -> torch.Tensor:
    """
    Coarse-cell CE loss (KMR surrogate).
    q: [B,D], y_cells: [B,M] (row-normalized), returns scalar
    """
    s = coarse_head(q)  # [B,M]
    logp = F.log_softmax(s / tau, dim=1)
    return -(y_cells * logp).sum(dim=1).mean() * (tau ** 2)


