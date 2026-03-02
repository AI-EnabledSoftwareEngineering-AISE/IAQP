import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from tqdm.auto import tqdm

# Optional ANN libs (install as needed)
try:
    import faiss
except Exception as e:
    faiss = None
try:
    import hnswlib
except Exception as e:
    hnswlib = None

# -------------------- Candidate pack builder --------------------
@torch.no_grad()
def build_pack_topC(T_batch: torch.Tensor,
                    X_all: torch.Tensor,
                    C: int) -> torch.Tensor:
    """
    Build candidate packs by cosine similarity in base space.
    Returns ids_b: [B,C]
    """
    # T_batch [B,D] and X_all [N,D] must be L2-normalized
    S = torch.matmul(T_batch, X_all.T)  # [B,N]
    C_eff = min(C, X_all.size(0))
    ids = torch.topk(S, k=C_eff, dim=1, largest=True, sorted=False).indices
    return ids.long()


@torch.no_grad()
def dist_build_pack_topC(q_b: torch.Tensor,             # [B, D] (L2n) on local device
                         X_shard: torch.Tensor,         # [N_shard, D] (L2n) on local device
                         shard_offset: int,             # int offset into the global bank
                         C: int) -> torch.Tensor:       # returns [B, C] global ids
    """
    EXACT Top-C across shards (DDP-safe, no recall drop)
    Each rank holds a shard X_shard = X_all[shard_start: shard_end]
    1) local topC on shard, 2) all_gather, 3) global topC merge.
    """
    import torch.distributed as dist
    B = q_b.size(0)
    
    # local similarities
    S_local = q_b @ X_shard.t()                         # [B, N_shard]
    # local topC
    C_loc = min(C, X_shard.size(0))
    v_loc, p_loc = torch.topk(S_local, k=C_loc, dim=1, largest=True, sorted=False)
    ids_loc = p_loc + shard_offset                      # [B, C_loc] global ids

    if not (dist.is_available() and dist.is_initialized()):
        # single GPU: done
        if C_loc < C:  # pad
            pad = C - C_loc
            ids_loc = torch.cat([ids_loc, torch.full((B, pad), -1, device=q_b.device, dtype=torch.long)], 1)
        return ids_loc

    # gather from all ranks
    world_size = dist.get_world_size()
    ids_list = [torch.empty_like(ids_loc) for _ in range(world_size)]
    val_list = [torch.empty_like(v_loc)   for _ in range(world_size)]
    dist.all_gather(ids_list, ids_loc)
    dist.all_gather(val_list, v_loc)

    # concat candidates
    ids_all = torch.cat(ids_list, dim=1)  # [B, world_size*C_loc]
    val_all = torch.cat(val_list, dim=1)  # [B, world_size*C_loc]

    # final global topC
    C_eff = min(C, ids_all.size(1))
    v_fin, p_fin = torch.topk(val_all, k=C_eff, dim=1, largest=True, sorted=False)
    bidx = torch.arange(B, device=q_b.device).unsqueeze(1).expand_as(p_fin)
    ids_fin = ids_all[bidx, p_fin]  # [B, C_eff]

    if C_eff < C:
        pad = C - C_eff
        ids_fin = torch.cat([ids_fin, torch.full((B, pad), -1, device=q_b.device, dtype=torch.long)], 1)

    return ids_fin

@torch.no_grad()
def build_pack_topC_streaming(T_batch: torch.Tensor,  # [B, D] L2n on device
                              X_all: torch.Tensor,    # [N, D] L2n on CPU or device
                              C: int,
                              x_batch: int = 100_000,
                              use_fp16: bool = True) -> torch.Tensor:
    """
    EXACT top-C without allocating [B, N].
    Streams over X_all in tiles of size x_batch and merges top-C.
    Handles both CPU and GPU X_all tensors for memory efficiency.
    """
    dev = T_batch.device
    B = T_batch.size(0)
    C = int(min(C, X_all.size(0)))

    # running best
    best_vals = torch.full((B, C), -1e9, device=dev, dtype=torch.float32)
    best_ids  = torch.full((B, C),   -1, device=dev, dtype=torch.long)

    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
    except: pass

    for xs in range(0, X_all.size(0), x_batch):
        xe = min(X_all.size(0), xs + x_batch)
        # Handle CPU X_all by streaming chunks to GPU
        if X_all.device.type == "cpu":
            X_tile = X_all[xs:xe].to(dev, non_blocking=True)  # [x_batch, D] CPU -> GPU
        else:
            X_tile = X_all[xs:xe]  # [x_batch, D] already on device

        with torch.amp.autocast('cuda', enabled=(use_fp16 and dev.type=="cuda")):
            S = T_batch @ X_tile.T  # [B, x_batch] half/TF32

        kk = min(C, S.size(1))
        v, p = torch.topk(S, k=kk, dim=1, largest=True, sorted=False)
        v = v.to(torch.float32)
        ids = (p + xs).to(torch.long)

        # pad if needed
        if kk < C:
            pad = C - kk
            v   = torch.cat([v,   torch.full((B, pad), -1e9, device=dev)], 1)
            ids = torch.cat([ids, torch.full((B, pad), -1,   device=dev, dtype=torch.long)], 1)

        both_v = torch.cat([best_vals, v  ], 1)
        both_i = torch.cat([best_ids,  ids], 1)
        topv, topp = torch.topk(both_v, k=C, dim=1, largest=True, sorted=False)
        row = torch.arange(B, device=dev).unsqueeze(1)
        best_vals = topv
        best_ids  = both_i[row, topp]

        del X_tile, S, v, p, ids, both_v, both_i, topv, topp

    return best_ids


@torch.no_grad()
def precompute_packs_topC_ddp(
        T_cpu: torch.Tensor,        # [N, D], on CPU, L2-normalized
        X_cpu: torch.Tensor,        # [Nimg, D], on CPU, L2-normalized
        C: int,
        chunk: int = 16384,         # texts per step (A100: 16–32k good)
        x_batch: int = 131072,      # images per tile (FP16: 128–256k; TF32: 64–128k)
        use_fp16: bool = True,
        ddp: bool = True,
        gather: str = "disk",       # "disk" (robust) or "nccl" (in-RAM gather)
        out_dir: str = "./packs_tmp",
        fname: str = "packs_topC.pt",
    ) -> torch.Tensor:
    import os, math
    from tqdm.auto import tqdm
    import torch.distributed as dist

    if ddp and not (dist.is_available() and dist.is_initialized()):
        ddp = False

    N, D = T_cpu.shape
    world = dist.get_world_size() if ddp else 1
    rank  = dist.get_rank()      if ddp else 0

    # shard queries
    per = (N + world - 1) // world
    qs = rank * per
    qe = min(N, qs + per)
    N_local = max(0, qe - qs)

    # nothing to do
    if N_local == 0:
        if ddp: dist.barrier()
        if rank == 0:
            # stitch if others wrote shards
            if gather == "disk":
                shards = []
                for r in range(world):
                    path = os.path.join(out_dir, f"{fname}.rank{r}.pt")
                    shards.append(torch.load(path, map_location="cpu"))
                ids_all = torch.cat(shards, 0)[:N].to(torch.int32)
                # save final file
                torch.save(ids_all, os.path.join(out_dir, fname))
                # (optional) cleanup shards
                for r in range(world):
                    try: os.remove(os.path.join(out_dir, f"{fname}.rank{r}.pt"))
                    except: pass
                return ids_all
            else:
                # nccl gather case handled below when others reach barrier
                return torch.empty(N, C, dtype=torch.int32)
        else:
            return torch.empty(0, C, dtype=torch.int32)

    os.makedirs(out_dir, exist_ok=True)

    # local CUDA device
    dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # speed knobs
    try: torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
    except: pass
    torch.backends.cuda.matmul.allow_tf32 = True

    # push local T slice to GPU in big chunks during loop (not all at once)
    # stream X tiles from CPU→GPU each inner step
    out_local = torch.empty((N_local, C), dtype=torch.int32)

    for s in tqdm(range(qs, qe, chunk), disable=(rank!=0), desc=f"Precompute r{rank}", unit="batch"):
        e = min(qe, s + chunk)
        Tb = T_cpu[s:e].to(dev, non_blocking=True)
        B  = Tb.size(0)

        best_vals = torch.full((B, C), -1e9, device=dev, dtype=torch.float32)
        best_ids  = torch.full((B, C),   -1, device=dev, dtype=torch.long)

        for xs in range(0, X_cpu.size(0), x_batch):
            xe = min(X_cpu.size(0), xs + x_batch)
            Xb = X_cpu[xs:xe].to(dev, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(use_fp16 and dev.type=="cuda")):
                S = Tb @ Xb.T                    # [B, x_batch]

            kk = min(C, S.size(1))
            v, p = torch.topk(S, k=kk, dim=1, largest=True, sorted=False)  # UNSORTED = faster
            v = v.to(torch.float32)
            ids = (p + xs).to(torch.long)

            if kk < C:
                pad = C - kk
                v   = torch.cat([v,   torch.full((B, pad), -1e9, device=dev)], 1)
                ids = torch.cat([ids, torch.full((B, pad), -1,   device=dev, dtype=torch.long)], 1)

            both_v = torch.cat([best_vals, v  ], 1)
            both_i = torch.cat([best_ids,  ids], 1)
            topv, topp = torch.topk(both_v, k=C, dim=1, largest=True, sorted=False)
            row = torch.arange(B, device=dev).unsqueeze(1)
            best_vals = topv
            best_ids  = both_i[row, topp]

            del Xb, S, v, p, ids, both_v, both_i, topv, topp

        out_local[s-qs:e-qs] = best_ids.to("cpu").to(torch.int32)
        del Tb, best_vals, best_ids
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # gather
    if ddp:
        dist.barrier()
        if gather == "disk":
            # write shard; rank 0 stitches later
            torch.save(out_local, os.path.join(out_dir, f"{fname}.rank{rank}.pt"))
            dist.barrier()
            if rank == 0:
                shards = []
                for r in range(world):
                    shards.append(torch.load(os.path.join(out_dir, f"{fname}.rank{r}.pt"), map_location="cpu"))
                ids_all = torch.cat(shards, 0)[:N].to(torch.int32)
                torch.save(ids_all, os.path.join(out_dir, fname))
                for r in range(world):
                    try: os.remove(os.path.join(out_dir, f"{fname}.rank{r}.pt"))
                    except: pass
                return ids_all
            else:
                return out_local  # not used
        else:
            # in-RAM gather (requires equal sizes; we already sharded evenly enough)
            tensor_list = [torch.empty_like(out_local) for _ in range(world)]
            dist.all_gather(tensor_list, out_local)
            if rank == 0:
                ids_all = torch.cat(tensor_list, 0)[:N].to(torch.int32)
                torch.save(ids_all, os.path.join(out_dir, fname))
                return ids_all
            else:
                return out_local
    else:
        torch.save(out_local, os.path.join(out_dir, fname))
        return out_local


@torch.no_grad()
def precompute_packs_topC(T_all: torch.Tensor,
                          X_all: torch.Tensor,
                          C: int,
                          chunk: int = 8192,
                          x_batch: int = 25000) -> torch.Tensor:
    """
    Precompute base-space top-C packs for every text using streaming approach.
    Returns ids_all [N,C] on CPU (torch.long).
    """
    N = T_all.size(0)
    device = T_all.device
    ids_chunks = []
    
    for s in tqdm(range(0, N, chunk), desc="Precompute packs", unit="chunk"):
        e = min(N, s + chunk)
        T_batch = T_all[s:e]  # [chunk, D]
        B = T_batch.size(0)
        
        # Initialize running top-C for this text batch
        best_vals = torch.full((B, C), -1e9, device=device, dtype=torch.float32)
        best_ids = torch.full((B, C), -1, device=device, dtype=torch.long)
        
        # Stream through image bank in chunks
        for xs in range(0, X_all.size(0), x_batch):
            xe = min(X_all.size(0), xs + x_batch)
            X_chunk = X_all[xs:xe]  # [x_batch, D]
            
            # Compute similarities for this chunk
            S_chunk = torch.matmul(T_batch, X_chunk.T)  # [B, x_batch]
            
            # Local top-C in this chunk
            C_local = min(C, S_chunk.size(1))
            local_vals, local_pos = torch.topk(S_chunk, k=C_local, dim=1, largest=True, sorted=True)
            local_ids = (local_pos + xs).to(torch.long)
            
            # Pad if C_local < C
            if C_local < C:
                pad = C - C_local
                pad_vals = torch.full((B, pad), -1e9, device=device, dtype=torch.float32)
                pad_ids = torch.full((B, pad), -1, device=device, dtype=torch.long)
                local_vals = torch.cat([local_vals, pad_vals], dim=1)
                local_ids = torch.cat([local_ids, pad_ids], dim=1)
            
            # Merge with running best
            vals_all = torch.cat([best_vals, local_vals], dim=1)  # [B, 2C]
            ids_all = torch.cat([best_ids, local_ids], dim=1)     # [B, 2C]
            topk_vals, topk_pos = torch.topk(vals_all, k=C, dim=1, largest=True, sorted=True)
            bidx = torch.arange(vals_all.size(0), device=device).unsqueeze(1).expand_as(topk_pos)
            best_vals = topk_vals
            best_ids = ids_all[bidx, topk_pos]
            
            # Clean up
            del X_chunk, S_chunk, local_vals, local_pos, local_ids, vals_all, ids_all, topk_vals, topk_pos, bidx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        ids_chunks.append(best_ids.detach().cpu())
        del T_batch, best_vals, best_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    ids_all = torch.vstack(ids_chunks)
    return ids_all



# -------------------- Training --------------------
class AnnHeadCache:
    """Cache ANN heads per (budget, text_idx) with LRU capacity to prevent memory explosion."""
    def __init__(self, enable: bool = True, khead: int = 256, device: str = "cuda", max_capacity: int = 100000):
        self.enable = enable
        self.khead = int(khead)
        self.device = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
        self.max_capacity = max_capacity
        self._store: Dict[Tuple[int,int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._access_order: List[Tuple[int,int]] = []  # LRU tracking

    def _evict_lru(self):
        """Evict least recently used entries when at capacity."""
        while len(self._store) >= self.max_capacity:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                if lru_key in self._store:
                    del self._store[lru_key]

    def clear(self):
        """Clear all cached ANN heads (e.g., at epoch start to avoid staleness)."""
        self._store.clear()
        self._access_order.clear()

    def get(self, B: int, text_indices: torch.Tensor):
        if not self.enable:
            return None, None, None
        ids_list, sims_list, miss_mask = [], [], []
        for tid in text_indices.tolist():
            key = (int(B), int(tid))
            if key in self._store:
                ids, sims = self._store[key]
                ids_list.append(ids)
                sims_list.append(sims)
                miss_mask.append(False)
                # Update LRU order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
            else:
                ids_list.append(None)
                sims_list.append(None)
                miss_mask.append(True)
        miss_mask = torch.tensor(miss_mask, device=self.device, dtype=torch.bool)
        return ids_list, sims_list, miss_mask

    def put_batch(self, B: int, text_indices: torch.Tensor,
                  ids_head: torch.Tensor, sims_head: torch.Tensor):
        if not self.enable:
            return
        
        # Evict if needed before adding new entries
        self._evict_lru()
        
        ptr = 0
        for tid in text_indices.tolist():
            key = (int(B), int(tid))
            self._store[key] = (ids_head[ptr].detach().clone(), sims_head[ptr].detach().clone())
            self._access_order.append(key)
            ptr += 1
def lambda_ann_for_budget(B: int, epoch: int = 1) -> float:
    """
    Budget-aware weight for ANN teacher with first-epoch fairness.
    Higher at small budgets; lower at large budgets.
    First epoch uses more conservative weights to avoid overfitting to bad packs.
    """
    # Base schedule: 0.9 at 10 → 0.6 at 100 (linear)
    base_lambda = max(0.6, 0.9 - (B - 10) * (0.3 / 90.0))
    
    # First epoch: clamp λ_ann lower for small budgets to avoid bad pack overfitting
    if epoch == 1 and B <= 30:
        # More conservative for small budgets in first epoch
        base_lambda = min(base_lambda, 0.7)
        if B <= 20:
            base_lambda = min(base_lambda, 0.6)
    
    return base_lambda



# -------------------- Evaluation --------------------
@torch.no_grad()
def project_np(model: nn.Module, T_np: np.ndarray, device: str = "cuda", batch: int = 65536) -> np.ndarray:
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    model.eval().to(dev)
    out = np.empty_like(T_np, dtype=np.float32)
    n = T_np.shape[0]
    for s in range(0, n, batch):
        e = min(n, s + batch)
        xb = torch.from_numpy(T_np[s:e]).to(dev).float()
        xb = F.normalize(xb, dim=1)
        yb = model(xb)
        out[s:e] = yb.detach().cpu().numpy().astype(np.float32)
    return out


@torch.no_grad()
def brute_force_topk_streaming(Q: np.ndarray,
                               X: np.ndarray,
                               k: int,
                               q_batch: int = 16384,
                               x_batch: int = 25000,
                               use_fp16: bool = False,
                               show_progress: bool = False) -> np.ndarray:
    """
    Exact top-k for large banks with bounded memory:
    - L2-normalized float32 inputs required
    - Streams over X in chunks and merges top-k
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = int(min(k, X.shape[0]))
    if k < 1: raise ValueError("k must be >= 1")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def merge_topk(vals_a, ids_a, vals_b, ids_b, k):
        va = torch.cat([vals_a, vals_b], 1)
        ia = torch.cat([ids_a,  ids_b],  1)
        v, pos = torch.topk(va, k=k, dim=1, largest=True, sorted=True)
        bidx = torch.arange(va.size(0), device=va.device).unsqueeze(1)
        i = ia[bidx, pos]
        return v, i

    out = []
    total_q_batches = (Q.shape[0] + q_batch - 1) // q_batch
    
    # Create progress bar for query batches
    if show_progress:
        from tqdm.auto import tqdm
        q_progress = tqdm(range(0, Q.shape[0], q_batch), 
                         desc="Computing brute_force_topk_streaming", unit="batch", 
                         total=total_q_batches)
    else:
        q_progress = range(0, Q.shape[0], q_batch)
    
    for s in q_progress:
        e = min(Q.shape[0], s + q_batch)
        Qt = torch.from_numpy(Q[s:e]).to(dev).contiguous()
        if use_fp16 and dev.type == "cuda":
            Qt = Qt.half()
            neg_inf = torch.tensor(-1e4, device=dev, dtype=Qt.dtype)
        else:
            neg_inf = torch.finfo(Qt.dtype).min

        B = Qt.size(0)
        best_vals = torch.full((B, k), neg_inf, device=dev, dtype=Qt.dtype)
        best_ids  = torch.full((B, k), -1, device=dev, dtype=torch.long)

        for xs in range(0, X.shape[0], x_batch):
            xe = min(X.shape[0], xs + x_batch)
            Xt = torch.from_numpy(X[xs:xe]).to(dev).contiguous()
            if use_fp16 and dev.type == "cuda":
                Xt = Xt.half()
            S = Qt @ Xt.T
            kk = min(k, S.size(1))
            v, p = torch.topk(S, k=kk, dim=1, largest=True, sorted=True)
            ids = (p + xs).to(torch.long)
            # pad if kk < k
            if kk < k:
                pad = k - kk
                v = torch.cat([v, torch.full((B, pad), neg_inf, device=dev, dtype=v.dtype)], 1)
                ids = torch.cat([ids, torch.full((B, pad), -1, device=dev, dtype=torch.long)], 1)
            best_vals, best_ids = merge_topk(best_vals, best_ids, v, ids, k)

            del Xt, S, v, p, ids

        out.append(best_ids.cpu().numpy().astype(np.int64))
        del Qt, best_vals, best_ids

    if show_progress:
        q_progress.close()
    
    return np.vstack(out)



def recall_at_k(pred_ids: np.ndarray, exact_ids: np.ndarray, k: int) -> float:
    """Recall@k: intersection of top-k predictions with top-k exact results."""
    assert pred_ids.ndim == 2 and exact_ids.ndim == 2
    k = min(k, pred_ids.shape[1], exact_ids.shape[1])
    total = 0
    for i in range(pred_ids.shape[0]):
        total += len(set(pred_ids[i, :k]).intersection(exact_ids[i, :k]))
    return total / float(k * pred_ids.shape[0])


def pair_hit_at_k(pred_ids: np.ndarray, gt_img_idx: np.ndarray, k: int) -> float:
    """
    Fraction of queries whose paired image appears in top-k.
    pred_ids: [N,K] int ids; gt_img_idx: [N] int
    """
    assert pred_ids.ndim == 2
    assert gt_img_idx.ndim == 1 and gt_img_idx.shape[0] == pred_ids.shape[0]
    k = max(1, min(k, pred_ids.shape[1]))
    P = pred_ids[:, :k]
    hits = (P == gt_img_idx[:, None])
    return float(hits.any(axis=1).mean())

# --- helpers --------------------------------------------------------
def get_split(data, split: str):
    """Return (X_img_train_bank, T_split, exact_idx_split, gt_img_idx_split)."""
    tr = data["train"]
    X_bank = l2n_np(tr["image_features"].astype(np.float32))  # always train images

    pk = data[split]
    T = l2n_np(pk["text_features"].astype(np.float32))
    exact_idx = pk["knn_indices"].astype(np.int64)            # already vs train bank

    # Optional paired mapping (kept if present)
    if "text_to_image" in pk:
        gt = np.asarray(pk["text_to_image"], dtype=np.int64)
    else:
        gt = np.arange(T.shape[0], dtype=np.int64)
    return X_bank, T, exact_idx, gt

# --- cache guardrails ------------------------------------------------
def validate_cache(data: dict):
    assert "train" in data, "cache missing 'train' split"
    assert "image_features" in data["train"] and "text_features" in data["train"], \
        "'train' must have 'image_features' and 'text_features'"
    for sp in ("val", "test"):
        if sp in data:
            assert "text_features" in data[sp] and "knn_indices" in data[sp], \
                f"'{sp}' must have 'text_features' and 'knn_indices'"
            assert data[sp].get("index_ref", "train") == "train", \
                f"'{sp}.index_ref' must be 'train' (all KNN vs train bank)"

# --- Fingerprint helpers ------------------------------------------------------

def _sha1(x: bytes) -> str:
    return hashlib.sha1(x).hexdigest()

def bank_fingerprint(X_bank: np.ndarray, R: "PCARSpace") -> str:
    """
    Stable id for a specific image bank under a specific rotation R.
    Changes if:
      - the bank content/order changes (sampled bytes),
      - (N, D) changes,
      - the rotation matrix W changes.
    """
    N, D = X_bank.shape
    buf = np.ascontiguousarray(X_bank, dtype=np.float32).tobytes()
    sample = buf[:10**6] + buf[-10**6:] if len(buf) > 2_000_000 else buf
    w_bytes = (np.ascontiguousarray(getattr(R, "W", None), dtype=np.float32).tobytes()
               if getattr(R, "W", None) is not None else b"")
    h = _sha1(sample + w_bytes + str((N, D)).encode())
    return f"N{N}_D{D}_{h[:12]}"


def _check_unit(name: str, A: np.ndarray):
    """Unit-norm check with detailed statistics."""
    n = np.linalg.norm(A, axis=1)
    print(f"[norm] {name}: mean={n.mean():.6f} min={n.min():.6f} max={n.max():.6f}")
    assert np.allclose(n, 1.0, atol=1e-4), f"{name} not unit normalized"


def suggested_nlist(N: int, hint: int) -> int:
    """Auto-scale ivf_nlist based on dataset size."""
    import math
    sug = max(16, int(4 * math.sqrt(N)))
    if hint and hint > 0:
        return min(sug, int(hint))
    return sug

def resolve_nlist(N: int, hint: int) -> int:
    # heuristic: 4*sqrt(N), clamped and guided by hint
    return max(16, min(int(4 * math.sqrt(N)), int(hint)))


# --- HNSW: get_or_build_hnsw --------------------------------------------------
def get_or_build_hnsw(X_R: np.ndarray,
                      indices_dir: str,
                      bank_fp: str,
                      M: int,
                      efC: int,
                      num_threads: int = 0):
    """
    Returns an hnswlib cosine index built on *rotated, unit-normalized* vectors X_R.
    Uses (bank_fp + build meta) to name files so different rotations/banks/params
    never collide. Rebuilds automatically on load failure or mismatch.
    """
    assert hnswlib is not None, "hnswlib not installed"
    os.makedirs(indices_dir, exist_ok=True)

    # Light sanity: expect unit-norm vectors for cosine distance semantics.
    nrm = np.linalg.norm(X_R, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-4):
        raise ValueError("get_or_build_hnsw: X_R must be L2-normalized.")

    N, D = X_R.shape
    meta = {
        "lib": "hnswlib",
        "space": "cosine",
        "M": int(M),
        "efC": int(efC),
        "dtype": "f32",
        "N": int(N),
        "D": int(D),
    }
    try:
        import hnswlib as _h
        meta["hnswlib_ver"] = getattr(_h, "__version__", "unknown")
    except Exception:
        pass

    fp = _sha1((bank_fp + str(meta)).encode())
    fname = f"hnsw_{bank_fp}_{fp[:8]}.bin"
    meta_name = f"hnsw_{bank_fp}_{fp[:8]}.json"
    path = os.path.join(indices_dir, fname)
    meta_path = os.path.join(indices_dir, meta_name)

    # Try to load existing index
    if os.path.exists(path):
        try:
            idx = hnswlib.Index(space="cosine", dim=D)
            idx.load_index(path)
            # Basic consistency check
            cur = idx.get_current_count()
            if cur != N:
                raise RuntimeError(f"HNSW ntotal mismatch: file has {cur}, expected {N}")
            return idx
        except Exception as e:
            # Corrupt or incompatible: rebuild
            try:
                os.remove(path)
            except Exception:
                pass

    # Build fresh
    idx = hnswlib.Index(space="cosine", dim=D)
    idx.init_index(max_elements=N, ef_construction=int(efC), M=int(M))
    
    # Set number of threads for HNSW
    if num_threads > 0:
        idx.set_num_threads(num_threads)
        print(f"    Using {num_threads} threads for HNSW")
    else:
        import os as os_module
        max_threads = os_module.cpu_count() or 1
        idx.set_num_threads(max_threads)
        print(f"    Using all {max_threads} CPU cores for HNSW")
    
    idx.add_items(X_R.astype(np.float32))
    try:
        idx.save_index(path)
        with open(meta_path, "w") as f:
            json.dump({**meta, "bank_fp": bank_fp, "file": fname}, f)
    except Exception:
        pass
    return idx

# --- IVF: get_or_build_ivf ----------------------------------------------------

def get_or_build_ivf(X_R: np.ndarray,
                     indices_dir: str,
                     bank_fp: str,
                     nlist_hint: int,
                     force_cpu: bool = False,
                     num_threads: int = 0):
    """
    Returns a FAISS IVFFlat(IP) index (moved to GPU if available), built on
    *rotated, unit-normalized* vectors X_R. CPU copy is cached to disk using a
    rotation-aware fingerprint so different banks/rotations/params don't collide.
    """
    assert faiss is not None, "faiss not installed"
    os.makedirs(indices_dir, exist_ok=True)

    nrm = np.linalg.norm(X_R, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-4):
        raise ValueError("get_or_build_ivf: X_R must be L2-normalized.")

    N, D = X_R.shape
    nlist = resolve_nlist(N, nlist_hint)
    print(f"Building IVF index with nlist={nlist}")

    meta = {
        "lib": "faiss",
        "type": "IVFFlat",
        "metric": "IP",
        "nlist": int(nlist),
        "dtype": "f32",
        "N": int(N),
        "D": int(D),
    }
    try:
        import faiss as _f
        meta["faiss_ver"] = getattr(_f, "__version__", "unknown")
    except Exception:
        pass

    fp = _sha1((bank_fp + str(meta)).encode())
    fname = f"ivf_{bank_fp}_{fp[:8]}.faiss"
    meta_name = f"ivf_{bank_fp}_{fp[:8]}.json"
    cpu_path = os.path.join(indices_dir, fname)
    meta_path = os.path.join(indices_dir, meta_name)

    def _build_cpu():
        quant = faiss.IndexFlatIP(D)
        cpu = faiss.IndexIVFFlat(quant, D, int(nlist), faiss.METRIC_INNER_PRODUCT)
        
        # Set number of threads for FAISS
        if num_threads > 0:
            faiss.omp_set_num_threads(num_threads)
            print(f"    Using {num_threads} threads for FAISS")
        else:
            import os as os_module
            max_threads = os_module.cpu_count() or 1
            faiss.omp_set_num_threads(max_threads)
            print(f"    Using all {max_threads} CPU cores for FAISS")
        
        Xf = X_R.astype(np.float32)
        
        # Use GPU for training if available, fallback to CPU
        try:
            if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
                # Create GPU index for training
                gpu_resource = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu)
                print(f"    Training IVF index on GPU...")
                gpu_index.train(Xf)
                # Move trained index back to CPU for adding
                cpu = faiss.index_gpu_to_cpu(gpu_index)
                print(f"    Training completed on GPU, moved back to CPU for adding")
            else:
                print(f"    Training IVF index on CPU (no GPU available)...")
                cpu.train(Xf)
        except Exception as e:
            print(f"    Warning: GPU training failed ({e}), falling back to CPU training")
            cpu.train(Xf)
        
        # Adding always happens on CPU
        print(f"    Adding vectors to IVF index on CPU...")
        cpu.add(Xf)
        return cpu

    # Load or build CPU index
    cpu_idx = None
    if os.path.exists(cpu_path):
        try:
            cpu_idx = faiss.read_index(cpu_path)
            # Basic consistency checks
            if cpu_idx.d != D or cpu_idx.ntotal != N:
                cpu_idx = None
                os.remove(cpu_path)
        except Exception:
            try:
                os.remove(cpu_path)
            except Exception:
                pass
            cpu_idx = None

    if cpu_idx is None:
        cpu_idx = _build_cpu()
        try:
            faiss.write_index(cpu_idx, cpu_path)
            with open(meta_path, "w") as f:
                json.dump({**meta, "bank_fp": bank_fp, "file": fname}, f)
        except Exception:
            pass

    # Move to GPU(s) if available, with memory management
    if not force_cpu:
        try:
            if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
                # Try to move to GPU, but fallback to CPU if memory issues
                try:
                    gpu_idx = faiss.index_cpu_to_all_gpus(cpu_idx)
                    return gpu_idx
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "alloc fail" in str(e).lower():
                        print(f"    Warning: GPU memory insufficient for IVF index, using CPU fallback")
                        return cpu_idx
                    else:
                        raise
        except Exception as e:
            print(f"    Warning: Could not move IVF index to GPU: {e}, using CPU fallback")
    else:
        print(f"    Using CPU evaluation as requested")
    return cpu_idx


# -------------------- Single GPU Utils --------------------
def setup_ivf_gpu_sharding(ivf_cpu_index, world_size: int = 1, local_rank: int = 0, force_cpu: bool = False):
    """
    Setup IVF index for single GPU training.
    Returns GPU index or CPU index if force_cpu=True.
    """
    if force_cpu:
        return ivf_cpu_index
    
    if faiss is None:
        return ivf_cpu_index
        
    try:
        if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            # Create GPU IVF for single GPU
            gpu_resource = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, ivf_cpu_index)
            return gpu_index
        else:
            return ivf_cpu_index
    except Exception as e:
        print(f"Warning: Could not move IVF to GPU: {e}")
        return ivf_cpu_index


# -------------------- Utils --------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------- Evaluation Utilities --------------------
def run_evaluation_section(index, backend_name, T_baseline_R, T_proj_R, exact_ref, exact_proj, 
                           eval_config, gt_img_idx, eval_table_func, results_dict, is_baseline=True):
    """
    Generic evaluation section runner to avoid code repetition.
    
    Args:
        index: The ANN index to evaluate
        backend_name: Name of the backend (e.g., "IVF", "HNSW", "IMI", "DiskANN", "NSG")
        T_baseline_R: Baseline (original) text features
        T_proj_R: Projected text features  
        exact_ref: Exact ground truth for baseline
        exact_proj: Exact ground truth for projected
        cfg: Configuration object
        gt_img_idx: Ground truth image indices
        eval_table_func: Function to run evaluation table
        results_dict: Dictionary to store results
        is_baseline: Whether this is baseline evaluation
    """
    if index is None:
        return
        
    if is_baseline:
        print(f"\n=== {backend_name} BASELINE (no projection) ===")
        # Convert backend name to lowercase and replace spaces with underscores
        backend_key = backend_name.lower().replace(' ', '_')
        rows = eval_table_func(index, backend_key, T_baseline_R, exact_ref,
                              list(range(10, 101, 10)), eval_config.eval_topk, gt_img_idx,
                              eval_ph=eval_config.eval_ph, eval_r5=eval_config.eval_r5)
        print(f"\nBackend: {backend_name} (baseline)")
        for B in sorted(rows.keys()):
            r, ph, qps, r5 = rows[B]
            bits = [f"R@{eval_config.eval_topk}={r:.4f}"]
            if eval_config.eval_ph: bits.append(f"PH@{eval_config.eval_topk}={ph:.4f}")
            bits.append(f"QPS={qps:.1f}")
            if eval_config.eval_r5: bits.append(f"R@5={r5:.4f}")
            print(f"  B={B:3d}  " + "  ".join(bits))
        results_dict[f"{backend_name}_original_features"] = {B: rows[B][0] for B in rows}
    else:
        print(f"\n=== {backend_name} PROJECTED ===")
        # Convert backend name to lowercase and replace spaces with underscores
        backend_key = backend_name.lower().replace(' ', '_')
        rows = eval_table_func(index, backend_key, T_proj_R, exact_proj,
                              list(range(10, 101, 10)), eval_config.eval_topk, gt_img_idx,
                              eval_ph=eval_config.eval_ph, eval_r5=eval_config.eval_r5)
        print(f"\nBackend: {backend_name} (projected)")
        for B in sorted(rows.keys()):
            r, ph, qps, r5 = rows[B]
            bits = [f"R@{eval_config.eval_topk}={r:.4f}"]
            if eval_config.eval_ph: bits.append(f"PH@{eval_config.eval_topk}={ph:.4f}")
            bits.append(f"QPS={qps:.1f}")
            if eval_config.eval_r5: bits.append(f"R@5={r5:.4f}")
            print(f"  B={B:3d}  " + "  ".join(bits))
        for B, (r, *_rest) in rows.items():
            assert 0.0 <= r <= 1.0, f"recall out of range at B={B}: {r}"
        results_dict[f"{backend_name}_projected_features"] = {B: rows[B][0] for B in rows}


def build_ann_indices(backend_config, eval_config, X_R_bank, bank_fp_eval):
    """
    Build all requested ANN indices for evaluation.
    
    Returns:
        dict: Dictionary containing built indices
    """
    # Import helper functions dynamically to avoid circular imports
    try:
        from .scripts.imi_helper import get_or_build_imi_ivf_ip
    except ImportError:
        get_or_build_imi_ivf_ip = None
    
    try:
        from .scripts.nsg_helper import get_or_build_nsg_ip
    except ImportError:
        get_or_build_nsg_ip = None
    
    try:
        from .scripts.diskann_helper import get_or_build_diskann
    except ImportError:
        get_or_build_diskann = None
    
    try:
        from .scripts.test_cuvs_hnsw import get_or_build_cuvs_hnsw
    except ImportError:
        get_or_build_cuvs_hnsw = None
    
    try:
        from .scripts.test_cuvs_cagara import get_or_build_cuvs_cagra
    except ImportError:
        get_or_build_cuvs_cagra = None
    
    try:
        import diskannpy as dap
    except ImportError:
        dap = None
    
    indices = {}
    
    # Set FAISS/HNSW threads
    if backend_config.num_threads and backend_config.num_threads > 0:
        try:
            import faiss
            faiss.omp_set_num_threads(backend_config.num_threads)
        except Exception:
            pass
    
    # Build indices based on eval_backend setting
    if eval_config.eval_backend in ["both", "hnsw"]:
        print("Building HNSW index for evaluation...")
        indices['hnsw'] = get_or_build_hnsw(X_R_bank, "indices/", bank_fp_eval, 
                                           M=eval_config.hnsw_M, efC=eval_config.hnsw_efC, 
                                           num_threads=backend_config.num_threads) if hnswlib else None
    
    if eval_config.eval_backend in ["both", "ivf"]:
        print("Building IVF index for evaluation...")
        indices['ivf'] = get_or_build_ivf(X_R_bank, "indices/", bank_fp_eval, 
                                         nlist_hint=backend_config.ivf_nlist, force_cpu=backend_config.force_cpu_eval, 
                                         num_threads=backend_config.num_threads) if faiss else None
    
    if eval_config.eval_backend in ["both", "imi"] and get_or_build_imi_ivf_ip:
        print("Building IMI index for evaluation...")
        indices['imi'] = get_or_build_imi_ivf_ip(X_R_bank, "indices/", bank_fp_eval, 
                                                ivf_nlist=backend_config.ivf_nlist, force_cpu=backend_config.force_cpu_eval, 
                                                num_threads=backend_config.num_threads) if faiss else None
    
    if eval_config.eval_backend in ["both", "diskann"] and get_or_build_diskann and dap:
        print("Building DiskANN index for evaluation...")
        indices['diskann'] = get_or_build_diskann(X_R_bank, "indices/", bank_fp_eval, 
                                                 graph_degree=eval_config.diskann_graph_degree, 
                                                 build_complexity=eval_config.diskann_build_complexity, 
                                                 num_threads=backend_config.num_threads)
    
    if eval_config.eval_backend in ["both", "nsg"] and get_or_build_nsg_ip:
        print("Building NSG index for evaluation...")
        indices['nsg'] = get_or_build_nsg_ip(X_R_bank, "indices/", bank_fp_eval, 
                                           R=eval_config.nsg_R, L=eval_config.nsg_L, 
                                           num_threads=backend_config.num_threads) if faiss else None
    
    if eval_config.eval_backend in ["both", "cuvs_hnsw"] and get_or_build_cuvs_hnsw:
        print("Building cuVS HNSW index for evaluation...")
        indices['cuvs_hnsw'] = get_or_build_cuvs_hnsw(X_R_bank, "indices/", bank_fp_eval,
                                                     M=eval_config.cuvs_hnsw_M, ef_construction=eval_config.cuvs_hnsw_efC,
                                                     mult_ef=eval_config.cuvs_hnsw_mult_ef, metric=eval_config.cuvs_hnsw_metric,
                                                     hierarchy=eval_config.cuvs_hnsw_hierarchy, force_cpu=backend_config.force_cpu_eval,
                                                     num_threads=backend_config.num_threads)
    
    if eval_config.eval_backend in ["both", "cuvs_cagra"] and get_or_build_cuvs_cagra:
        print("Building cuVS CAGRA index for evaluation...")
        indices['cuvs_cagra'] = get_or_build_cuvs_cagra(X_R_bank, "indices/", bank_fp_eval,
                                                       build_algo=backend_config.cuvs_cagra_build_algo,
                                                       metric=backend_config.cuvs_cagra_metric, force_cpu=backend_config.force_cpu_eval,
                                                       num_threads=backend_config.num_threads, Q_original=None, k=100, 
                                                       max_build_trials=3)
    
    print("✅ Built ANN indices for evaluation finished")
    return indices




def l2n_np(x: np.ndarray, eps=1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

# -------------------- Precompute functionality --------------------
def make_cell_target_with_spill(gt_img_ids: np.ndarray, labels: np.ndarray, spill_cells: np.ndarray, M: int, eps: float = 0.05) -> np.ndarray:
    """
    Create sparse target over regions with spill cells for orthogonality.
    Adds small epsilon mass to spill regions to improve routing robustness.
    """
    # Base target: counts per region among the ground-truth top-K
    reg_ids, counts = np.unique(labels[gt_img_ids], return_counts=True)
    y = np.zeros(M, dtype=np.float32)
    y[reg_ids] = counts.astype(np.float32)
    base_sum = y.sum()
    if base_sum > 0:
        y /= base_sum
    
    # Spill: add eps mass distributed over spill cells of positives
    s_ids = np.unique(spill_cells[gt_img_ids])
    if s_ids.size > 0:
        y[s_ids] += eps / float(s_ids.size)
    
    # Renormalize (cap to avoid exploding)
    y = y / (y.sum() + 1e-12)
    return y.astype(np.float32)


def pick_spill_cells(X_R: np.ndarray, Cents: np.ndarray, labels: np.ndarray, Rhat_bank: np.ndarray, 
                    topL: int = 8, beta: float = 0.1, device: str = "cuda") -> np.ndarray:
    """
    For each image i, pick a spill centroid c' that (i) keeps residual small
    and (ii) is as orthogonal as possible to the primary residual r.
    Cost: J = (r'·r_hat)^2 + beta * ||r'||^2. Returns int64 [N].
    """
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    Xt = torch.from_numpy(X_R).to(device if use_cuda else "cpu")
    Ct = torch.from_numpy(Cents).to(Xt.device)
    labt = torch.from_numpy(labels).to(Xt.device)
    rhat = torch.from_numpy(Rhat_bank).to(Xt.device)  # [N,D], unit residual

    N, D = Xt.shape
    spill = torch.empty(N, dtype=torch.long, device=Xt.device)

    # Preselect topL nearest centroids per point to avoid O(N*M)
    # Use cosine (dot) as fast proxy; both Xt and Ct are L2n already
    with torch.no_grad():
        # Batched to save memory
        bs = 8192
        for s in range(0, N, bs):
            e = min(N, s+bs)
            Xb = Xt[s:e]                              # [b,D]
            dots = Xb @ Ct.t()                        # [b,M]
            _, idxL = dots.topk(k=min(topL, Ct.size(0)), dim=1, largest=True)
            # Compute cost per candidate in idxL
            Ccand = Ct[idxL]                          # [b,topL,D]
            rprime = Xb.unsqueeze(1) - Ccand          # [b,topL,D]
            # ||r'||^2
            rpn2 = (rprime * rprime).sum(-1)          # [b,topL]
            # (r'·r_hat)^2 - use einsum to avoid broadcasting issues
            proj2 = torch.einsum('bld,bd->bl', rprime, rhat[s:e]).pow(2)  # [b,topL]
            J = proj2 + beta * rpn2                   # [b,topL]
            # Do not allow picking the primary cell again
            prim = labt[s:e].unsqueeze(1)             # [b,1]
            J = J.masked_fill(idxL.eq(prim), float('inf'))
            jmin = J.argmin(dim=1)
            spill[s:e] = idxL[torch.arange(idxL.size(0), device=idxL.device), jmin]
    return spill.detach().cpu().numpy().astype(np.int64)


def _kpp_init(X_t: torch.Tensor, k: int) -> torch.Tensor:
    """KMeans++ init on device; X_t is [N,D] L2-normalized."""
    N, D = X_t.shape
    dev = X_t.device
    idx0 = torch.randint(0, N, (1,), device=dev)
    cents = [X_t[idx0]]
    d2 = torch.full((N,), float("inf"), device=dev)
    
    while len(cents) < k:
        # Get the last centroid
        last_cents = cents[-1]  # [1, D]
        c = last_cents.squeeze(0)  # [D]
        
        # Compute cosine similarity: X_t @ c = [N,D] @ [D] = [N]
        sim = X_t @ c  # [N] - cosine similarities
        d2 = torch.minimum(d2, (1 - sim).clamp_min_(0))  # cosine→(1-sim) proxy
        probs = (d2 / (d2.sum() + 1e-12)).clamp_min_(1e-12)
        
        # Add exactly one new centroid
        new_idx = torch.multinomial(probs, num_samples=1, replacement=False)
        cents.append(X_t[new_idx])
    
    return torch.vstack(cents[:k]).contiguous()

def gpu_kmeans_fast(
    X: np.ndarray,
                    k: int,
                    device: str = "cuda",
                    max_iter: int = 50,
                    batch_size: int = 16384,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Large-N K-Means with true streaming over X.

    - Keeps the full dataset on CPU.
    - Moves only minibatches to GPU for assignment.
    - Centroids, per-cluster sums, and counts live on GPU.
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dev = torch.device(device if use_cuda else "cpu")

    # Keep X on CPU
    X_cpu = torch.from_numpy(X.astype(np.float32, copy=False))  # [N, D] on CPU
    N, D = X_cpu.shape

    # --- Initialization ---
    if use_cuda:
        # Random init from CPU (no full upload)
        perm = torch.randperm(N)[:k]
        init_cents = X_cpu[perm].to(dev, non_blocking=True)  # [k, D] on GPU
        cents = F.normalize(init_cents, dim=1)
        del init_cents
    else:
        # CPU path: reuse previous KMeans++ init semantics
        X_small = X_cpu.to(dev)
        cents = _kpp_init(X_small, k)
        cents = F.normalize(cents, dim=1)
        del X_small

    prev_cents = cents.clone()

    labels_cpu = torch.empty(N, dtype=torch.long)  # labels on CPU

    print(f"    Starting streaming K-means with {N:,} points, {k} clusters...")

    for it in range(max_iter):
        # Reset accumulators on device
        sums = torch.zeros(k, D, device=dev)
        counts = torch.zeros(k, device=dev)

        # E-step + accumulate for M-step
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            xb = X_cpu[s:e].to(dev, non_blocking=True)  # [b, D]

            sims = xb @ cents.T                         # [b, k]
            lbl = sims.argmax(dim=1)                    # [b]

            sums.index_add_(0, lbl, xb)
            counts.index_add_(0, lbl, torch.ones_like(lbl, dtype=torch.float32, device=dev))

            labels_cpu[s:e] = lbl.cpu()

            del xb, sims, lbl
            if use_cuda:
                torch.cuda.empty_cache()

        # Recompute centroids
        counts = counts.clamp_min_(1.0)
        cents = sums / counts.unsqueeze(1)
        cents = F.normalize(cents, dim=1)

        # Convergence check
        with torch.no_grad():
            shift = (1.0 - (cents * prev_cents).sum(1)).abs().mean().item()

        if shift < tol:
            print(f"    ✓ Converged @ iter {it+1}, mean angular shift={shift:.3e}")
            break

        prev_cents = cents.clone()

        if (it + 1) % 10 == 0 or it == 0:
            print(f"    iter {it+1}/{max_iter}  shift={shift:.3e}")

    # Optional: fresh labels pass with final centroids
    labels_cpu_final = torch.empty(N, dtype=torch.long)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        xb = X_cpu[s:e].to(dev, non_blocking=True)
        sims = xb @ cents.T
        lbl = sims.argmax(dim=1)
        labels_cpu_final[s:e] = lbl.cpu()
        del xb, sims, lbl
        if use_cuda:
            torch.cuda.empty_cache()

    cents_np = cents.detach().cpu().numpy().astype(np.float32)
    labels_np = labels_cpu_final.numpy().astype(np.int64)
    return cents_np, labels_np


def precompute_regions_and_residuals(X_train_R: np.ndarray, 
                                   exact_idx_train: np.ndarray,
                                   M: int = 4096,
                                   save_dir: str = "/home/hamed/projects/SPIN/adapter/data/laion-10M/precompute",
                                   device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """
    One-time precompute (offline, small):
    - Build generic regionization over the base vectors (GPU K-means)
    - Return only centroids and labels (no dense y_cells_per_text matrix)
    - Cell targets built on-the-fly per batch to avoid 160GB allocation
    """
    print(f"Precomputing regions and cell targets (M={M})...")
    
    # 1. Build regions using GPU-accelerated K-means
    print("  Step 1/2: Building regions with GPU K-means...")
    
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        # Fast GPU K-means
        Cents, labels = gpu_kmeans_fast(X_train_R, M, device=device, max_iter=50)
        Cents = l2n_np(Cents.astype(np.float32))  # L2-normalize centroids
        print(f"    ✓ Created {M} regions on GPU")
    else:
        # CPU fallback
        try:
            from sklearn.cluster import MiniBatchKMeans
            print(f"    Using CPU K-means with {M} clusters on {X_train_R.shape[0]:,} samples...")
            kmeans = MiniBatchKMeans(n_clusters=M, batch_size=32768, n_init="auto")
            labels = kmeans.fit_predict(X_train_R)
            Cents = l2n_np(kmeans.cluster_centers_.astype(np.float32))
            print(f"    ✓ Created {M} regions on CPU")
        except ImportError:
            raise ImportError("scikit-learn is required for CPU K-means. Install with: pip install scikit-learn")
    
    # 2. DO NOT allocate y_cells_per_text here - build on-the-fly per batch
    print("  Step 2/2: Skipping dense matrix allocation (will build on-the-fly)...")
    
    # 3. Sanity checks
    # print("  Running sanity checks...")
    
    # # Guard assumptions: L2-normalization
    # assert np.allclose(np.linalg.norm(X_train_R, axis=1), 1, atol=1e-5), "X_train_R not L2-normalized"
    # assert np.allclose(np.linalg.norm(Cents, axis=1), 1, atol=1e-5), "Cents not L2-normalized"
    
    # # Target mass sanity
    # y = y_cells_per_text  # [N_texts, M]
    # assert np.allclose(y.sum(1), 1.0, atol=1e-6), "y_cells_per_text not normalized"
    # assert (y >= -1e-7).all(), "y_cells_per_text has negative values"
    # print("    ✓ Target mass sanity check passed")
    
    # # No saving - everything computed on-the-fly
    # print(f"✓ Computed regions and cell targets on-the-fly")
    # print(f"  - Cents: {Cents.shape} (region centroids)")
    # print(f"  - labels: {labels.shape} (image->cell assignments)")
    
    return Cents, labels.astype(np.int32)


def load_precomputed_data(save_dir: str = "data/laion-10M/precompute") -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed regions and cell targets."""
    Cents = np.load(f"{save_dir}/Cents.npy").astype(np.float32)
    y_cells_per_text = np.load(f"{save_dir}/y_cells_per_text.npy").astype(np.float32)
    
    print(f"✓ Loaded precomputed data from {save_dir}/")
    print(f"  - Cents: {Cents.shape} (region centroids)")
    print(f"  - y_cells_per_text: {y_cells_per_text.shape} (cell targets)")
    
    return Cents, y_cells_per_text


def exact_teacher_in_pack(cand_ids_b: torch.Tensor,
                          exact_ids_for_batch: torch.Tensor) -> torch.Tensor:
    """Vectorized exact-topK teacher on the pack."""
    B, C = cand_ids_b.shape
    K = exact_ids_for_batch.shape[1]
    match = cand_ids_b.unsqueeze(2).eq(exact_ids_for_batch.unsqueeze(1))  # [B,C,K]
    pos = match.any(dim=2)                                                # [B,C]
    P = torch.zeros(B, C, device=cand_ids_b.device, dtype=torch.float32)
    counts = pos.sum(dim=1, keepdim=True).clamp_min(1)
    P[pos] = 1.0
    P = P / counts
    return P


def exact_teacher_in_pack_ranked(cand_ids_b: torch.Tensor,
                                 exact_ids_b: torch.Tensor,
                                 gamma: float = 0.15) -> torch.Tensor:
    """Rank-weighted exact teacher on the pack."""
    # cand_ids_b: [B,C], exact_ids_b: [B,K]
    B, C = cand_ids_b.shape
    K = exact_ids_b.shape[1]
    pos = torch.arange(K, device=cand_ids_b.device).view(1,1,K).expand(B,C,K)
    match = cand_ids_b.unsqueeze(2).eq(exact_ids_b.unsqueeze(1))             # [B,C,K]
    ranks = torch.where(match, pos, torch.full_like(pos, K)).min(dim=2).values.float()  # [B,C]
    w = torch.exp(-gamma * ranks) * (ranks < K).float()
    return w / (w.sum(dim=1, keepdim=True) + 1e-12)


def pack_size_for(B: int, Cmax: int = 256) -> int:
    """Budget-aware pack size: smaller packs at low B."""
    # smaller packs at low B; keeps compute down and supervision focused
    return int(min(Cmax, max(96, 6 * B)))


@torch.no_grad()
def _fill_scores_from_head(ids_head: torch.Tensor,  # [B,Khead] long (device)
                           sims_head: torch.Tensor, # [B,Khead] float (device)
                           cand_ids_b: torch.Tensor # [B,C] long (device)
                           ) -> torch.Tensor:
    """
    Memory-efficient mapping from retrieved (ids_head, sims_head) to scores for cand_ids_b.
    Returns [B,C] filled where a match exists, else -1e9.
    
    Avoids creating [B,Khead,C] tensors by processing matches incrementally.
    """
    B, C = cand_ids_b.shape
    Khead = ids_head.shape[1]
    
    # Initialize scores with -inf
    scores = torch.full((B, C), -1e9, device=cand_ids_b.device, dtype=sims_head.dtype)
    
    # Process matches without creating full [B,Khead,C] tensor
    # For each query, find matches between ids_head and cand_ids_b
    for k in range(Khead):
        # ids_head[:, k] is [B], cand_ids_b is [B, C]
        # Find where ids_head[:, k] matches any cand_ids_b[b, :]
        ids_k = ids_head[:, k].unsqueeze(1)  # [B, 1]
        matches = (ids_k == cand_ids_b)  # [B, C] bool
        sims_k = sims_head[:, k].unsqueeze(1)  # [B, 1]
        # Update scores where matches occur (take max to handle multiple matches)
        scores = torch.where(matches, torch.maximum(scores, sims_k), scores)
    
    return scores




# -------------------- Rotation (PCARSpace) --------------------
class PCARSpace:
    """
    Rotation-only transform (orthonormal rows). Fit on images, apply to both images + queries.
    Do NOT subtract mean at inference (hurts cosine).
    This version streams on GPU and uses covariance eigendecomp (D×D), not giant SVD (N×D).
    """
    def __init__(self, d_keep: Optional[int] = None, center_for_fit: bool = True,
                 device: str = "cuda", chunk: int = 262144, enable_tf32: bool = True):
        self.W = None          # [d_keep, D], orthonormal rows
        self.mu = None         # [1, D] or None
        self.d_keep = d_keep
        self.center_for_fit = center_for_fit
        self.device = device
        self.chunk = int(chunk)
        self.enable_tf32 = enable_tf32

    def fit(self, X_img: np.ndarray):
        X = np.asarray(X_img, dtype=np.float32, order="C")
        N, D = X.shape

        use_cuda = (str(self.device).startswith("cuda") and torch.cuda.is_available())
        if use_cuda:
            # Fast path: streamed GPU covariance
            if self.enable_tf32:
                try:
                    torch.set_float32_matmul_precision("high")  # enables TF32 matmul on Ampere+
                except Exception:
                    pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            dev = torch.device(self.device)
            # ---------- Pass 1: mean ----------
            if self.center_for_fit:
                s = torch.zeros(D, device=dev, dtype=torch.float64)
                n_tot = 0
                for i in range(0, N, self.chunk):
                    j = min(N, i + self.chunk)
                    xb = torch.as_tensor(X[i:j], device=dev)
                    s += xb.sum(dim=0, dtype=torch.float64)
                    n_tot += (j - i)
                mu = (s / float(n_tot)).to(torch.float32)  # [D]
                self.mu = mu.detach().cpu().unsqueeze(0).numpy()
            else:
                self.mu = None
                mu = None

            # ---------- Pass 2: Gram matrix G = (X - mu)^T (X - mu) ----------
            G = torch.zeros(D, D, device=dev, dtype=torch.float64)
            for i in range(0, N, self.chunk):
                j = min(N, i + self.chunk)
                xb = torch.as_tensor(X[i:j], device=dev)  # [b,D], f32
                if mu is not None:
                    xb = xb - mu
                # matmul in f32; accumulate in f64 for stability
                G += (xb.t().matmul(xb)).to(torch.float64)

            # Convert to covariance (optional scaling)
            # Using Gram is enough for eigenvectors; scaling doesn't change directions.
            G = G.to(torch.float32)

            # ---------- Eigen on D×D (tiny) ----------
            # G is symmetric PSD; eigh is ideal
            evals, evecs = torch.linalg.eigh(G)  # evals ascending
            idx = torch.argsort(evals, descending=True)
            V = evecs[:, idx]                    # [D,D] columns are principal directions

            # Keep top-d if requested
            if self.d_keep is not None and self.d_keep > 0:
                V = V[:, :self.d_keep]          # [D, d_keep]

            # Your code expects rows = directions (like V^T from SVD)
            self.W = V.t().contiguous().detach().cpu().numpy().astype(np.float32)  # [d_keep, D]

            # Orthonormal sanity (optional)
            # WW^T ≈ I
            # assert np.allclose(self.W @ self.W.T, np.eye(self.W.shape[0], dtype=np.float32), atol=1e-3)

            return self

        # -------- CPU fallback: original NumPy SVD (unchanged semantics) --------
        Xc = X - X.mean(0, keepdims=True) if self.center_for_fit else X
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        if self.d_keep is not None and self.d_keep > 0:
            Vt = Vt[: self.d_keep]
        self.W = Vt.astype(np.float32)
        self.mu = X.mean(0, keepdims=True).astype(np.float32) if self.center_for_fit else None
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated PCA transformation with batching for large datasets.
        """
        import torch
        import torch.nn.functional as F
        
        if self.device == "cpu":
            # CPU fallback
            Zr = Z.astype(np.float32) @ self.W.T
            return l2n_np(Zr).astype(np.float32)
        
        # GPU-accelerated transformation
        Z_tensor = torch.from_numpy(Z.astype(np.float32)).to(self.device)
        W_tensor = torch.from_numpy(self.W.astype(np.float32)).to(self.device)
        
        # Batch processing for large datasets
        batch_size = min(self.chunk, Z.shape[0])
        results = []
        
        for i in range(0, Z.shape[0], batch_size):
            end_idx = min(i + batch_size, Z.shape[0])
            Z_batch = Z_tensor[i:end_idx]
            
            # Matrix multiplication: Z @ W.T
            Zr_batch = Z_batch @ W_tensor.T
            
            # L2 normalization using PyTorch (faster than numpy)
            Zr_normalized = F.normalize(Zr_batch, p=2, dim=1)
            
            results.append(Zr_normalized.cpu().numpy())
        
        # Concatenate results
        Zr = np.concatenate(results, axis=0).astype(np.float32)
        return Zr


# -------------------- Model --------------------
class ResidualProjector(nn.Module):
    """
    Query-side residual: f(t) = norm(t + alpha * g(t))
    """
    def __init__(self, dim: int, hidden: int, alpha: float = 0.25):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, dim),
        )
        self.alpha = alpha

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t + self.alpha * self.g(t), dim=1)


class CoarseCellHead(nn.Module):
    """
    Coarse-cell probability head: learns to put probability mass on regions 
    that contain the query's true neighbors.
    """
    def __init__(self, dim: int, M: int):
        super().__init__()
        # simple linear layer initialized to cosine with centroids
        self.C = nn.Parameter(torch.empty(M, dim))  # will init with Cents
        nn.init.normal_(self.C, std=0.02)
    
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: [B, D] (L2-normalized), returns [B, M] cosine scores"""
        C = F.normalize(self.C, dim=1)
        return q @ C.T  # [B, M], cosine scores


class QuantityHead(nn.Module):
    """
    Predicts how much budget to spend (optional).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128), 
            nn.ReLU(True),
            nn.Linear(128, 1)
        )
    
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: [B, D], returns [B] budget predictions"""
        return self.mlp(q).squeeze(-1)


# -------------------- Distributed Utils --------------------
def init_distributed():
    """Initialize distributed training"""
    import torch.distributed as dist
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Cleanup distributed training"""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
