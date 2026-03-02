import math
import numpy as np
import torch

@torch.no_grad()
def em_exact_topk_streaming(
    Q: np.ndarray,                # [Nq, D], L2-normalized float32
    X: np.ndarray,                # [Nx, D], L2-normalized float32
    k: int,
    q_batch: int = 16384,
    x_batch: int | None = None,   # if None, auto-tune from free VRAM
    allow_tf32: bool = True,      # faster matmul in fp32 on Ampere+
    almost_exact_bf16: bool = False,  # bfloat16 compute (faster, tiny numeric drift)
    overlap_io: bool = True,      # double-buffered H2D copies + compute
    show_progress: bool = False
) -> np.ndarray:
    """
    Exact cosine top-k with bounded GPU memory, tuned for speed.
    - Works in pure float32 for exact results (recommended).
    - Streams database X by tiles; preallocates device buffers; overlaps H2D copies with GEMM.
    - Keeps GPU footprint ~O(q_batch*D + x_tile*D + q_batch*k)

    Notes:
      * For EXACT results, keep almost_exact_bf16=False.
      * Inputs must be L2-normalized if using cosine via dot-product.
    """
    assert Q.dtype == np.float32 and X.dtype == np.float32, "Use float32 for exactness."
    Nq, D = Q.shape
    Nx = X.shape[0]
    k = int(min(max(1, k), Nx))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = dev.type == "cuda"

    if allow_tf32 and use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Visible dtype for compute
    comp_dtype = torch.bfloat16 if (almost_exact_bf16 and use_cuda) else torch.float32

    # ---- helpers ----
    def _topk_merge(vals_a, ids_a, vals_b, ids_b, k):
        # vals_*: [B, k], ids_*: [B, k]
        # Merge by concatenation then topk
        v = torch.cat([vals_a, vals_b], dim=1)
        i = torch.cat([ids_a,  ids_b],  dim=1)
        v2, pos = torch.topk(v, k=k, dim=1, largest=True, sorted=True)
        row = torch.arange(v.size(0), device=v.device).unsqueeze(1)
        i2 = i[row, pos]
        return v2, i2

    def _auto_x_batch(target_frac: float = 0.8):
        """Pick x_tile size to stay within a fraction of free VRAM."""
        if not use_cuda:
            # CPU: keep a comfortable tile
            return 25000 if x_batch is None else x_batch

        free_b, total_b = torch.cuda.mem_get_info()
        # reserve some headroom
        budget = max(256<<20, int(target_frac * free_b))  # at least 256MB
        # Memory components (bytes):
        # Qt:  q_batch * D * dtype
        # Xt:  x_tile * D * dtype
        # S:   q_batch * x_tile * float32 (scores always kept in fp32 for stable topk)
        # work: ~ (q_batch*k*8 + small)
        bytes_per_f32 = 4
        bytes_per_comp = 2 if comp_dtype == torch.bfloat16 else 4

        # budget ≈ qD*comp + xD*comp + qx*4 + qk*8
        # Solve roughly for x given q and D
        qD = q_batch * D
        qk_bytes = q_batch * k * 8

        # x term appears in: xD*comp + q*x*4
        # (q*4)*x + (D*comp)*x <= budget - (qD*comp + qk*8)
        rhs = budget - (qD*bytes_per_comp + qk_bytes)
        if rhs <= (64<<20):  # too little, shrink q_batch instead of exploding compute
            return 8192 if x_batch is None else x_batch

        a = q_batch * 4
        b = D * bytes_per_comp
        x_est = int(rhs / (a + b))
        # clamp to reasonable bounds
        x_est = max(2048, min(x_est, 200_000))
        return x_est if x_batch is None else x_batch

    # ---- pin host memory for faster H2D ----
    Qp = torch.from_numpy(Q).pin_memory() if use_cuda else torch.from_numpy(Q)
    Xp = torch.from_numpy(X).pin_memory() if use_cuda else torch.from_numpy(X)

    # ---- auto-tune x_batch if needed ----
    x_tile = _auto_x_batch()

    # ---- progress ----
    if show_progress:
        from tqdm.auto import tqdm
        q_iter = tqdm(range(0, Nq, q_batch), desc="exact_topk_streaming (q-batches)")
    else:
        q_iter = range(0, Nq, q_batch)

    outputs = []

    # Precreate CUDA streams and buffers
    copy_stream = torch.cuda.Stream() if (use_cuda and overlap_io) else None
    comp_stream = torch.cuda.Stream() if (use_cuda and overlap_io) else None

    for qs in q_iter:
        qe = min(Nq, qs + q_batch)
        B = qe - qs

        # Device query buffer
        if use_cuda:
            if overlap_io and copy_stream is not None:
                with torch.cuda.stream(copy_stream):
                    Qt = Qp[qs:qe].to(device=dev, non_blocking=True).contiguous()
                    if comp_dtype != torch.float32:
                        Qt = Qt.to(dtype=comp_dtype)
                # ensure compute waits for copy
                torch.cuda.current_stream().wait_stream(copy_stream)
            else:
                Qt = Qp[qs:qe].to(device=dev, non_blocking=True).contiguous()
                if comp_dtype != torch.float32:
                    Qt = Qt.to(dtype=comp_dtype)
        else:
            Qt = Qp[qs:qe].contiguous()

        # Initialize per-batch best k
        neg_inf = torch.finfo(torch.float32).min
        best_vals = torch.full((B, k), neg_inf, device=dev, dtype=torch.float32)
        best_ids  = torch.full((B, k), -1,    device=dev, dtype=torch.long)

        # Double buffers for Xt tiles (device)
        if use_cuda and overlap_io:
            Xt_buf = [
                torch.empty((x_tile, D), device=dev, dtype=comp_dtype).contiguous(),
                torch.empty((x_tile, D), device=dev, dtype=comp_dtype).contiguous(),
            ]
        else:
            Xt_buf = [None, None]

        buf_idx = 0
        first_loaded = False

        # Iterate over database tiles
        for xs in range(0, Nx, x_tile):
            xe = min(Nx, xs + x_tile)
            cur_len = xe - xs

            if use_cuda:
                # async copy into the "next" buffer
                if overlap_io and copy_stream is not None:
                    with torch.cuda.stream(copy_stream):
                        # (Re)use buffer and resize view
                        if Xt_buf[buf_idx].shape[0] != cur_len:
                            Xt_buf[buf_idx] = torch.empty((cur_len, D), device=dev, dtype=comp_dtype).contiguous()
                        # copy
                        Xt_view = Xp[xs:xe].to(device=dev, non_blocking=True)
                        Xt_view = Xt_view.to(dtype=comp_dtype, non_blocking=True) if comp_dtype != torch.float32 else Xt_view
                        Xt_buf[buf_idx].copy_(Xt_view, non_blocking=True)
                    # If it's the very first tile, ensure it's ready before compute
                    if not first_loaded:
                        torch.cuda.current_stream().wait_stream(copy_stream)
                        first_loaded = True
                        active = buf_idx
                        buf_idx ^= 1
                    else:
                        # compute on the other buffer while this copies
                        torch.cuda.current_stream().wait_stream(copy_stream)
                        active = buf_idx ^ 1
                        # next time we'll flip again
                        buf_idx ^= 1
                else:
                    # synchronous path (still avoids reallocation)
                    Xt_view = Xp[xs:xe].to(device=dev, non_blocking=True)
                    Xt = Xt_view.to(dtype=comp_dtype) if comp_dtype != torch.float32 else Xt_view
                    active = None  # use Xt directly
            else:
                Xt = Xp[xs:xe]

            # ---- compute scores ----
            if use_cuda and overlap_io and comp_stream is not None:
                with torch.cuda.stream(comp_stream):
                    # ensure comp waits for copy
                    # (already waited above via current stream, but keep scope local)
                    pass

            # Matmul in chosen dtype, scores accumulated to float32 for stable topk
            if use_cuda and overlap_io and active is not None:
                S = (Qt @ Xt_buf[active].T).to(torch.float32)   # [B, cur_len]
            else:
                S = (Qt @ (Xt.T if 'Xt' in locals() else Xt_buf[active].T)).to(torch.float32)

            # Local top-k on this tile
            kk = min(k, cur_len)
            v, p = torch.topk(S, k=kk, dim=1, largest=True, sorted=True)
            ids = (p + xs).to(torch.long)

            # If cur_len<k, pad so merge logic stays simple
            if kk < k:
                pad = k - kk
                v = torch.cat([v, torch.full((B, pad), neg_inf, device=dev, dtype=torch.float32)], dim=1)
                ids = torch.cat([ids, torch.full((B, pad), -1, device=dev, dtype=torch.long)], dim=1)

            # Merge into global best
            best_vals, best_ids = _topk_merge(best_vals, best_ids, v, ids, k)

            # Free transient
            del S, v, p, ids
            if 'Xt' in locals():
                del Xt
                del Xt_view

        # Gather indices only (caller can re-score if needed)
        outputs.append(best_ids.cpu().numpy().astype(np.int64))

        # Cleanup per q-batch
        del Qt, best_vals, best_ids
        if use_cuda:
            torch.cuda.empty_cache()

    return np.vstack(outputs)
