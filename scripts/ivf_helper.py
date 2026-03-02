# IVF + IVFFlat (IP) search helper with memory-safe batch processing.
# Clean wrapper around FAISS IVF search with GPU memory fallback.

import time
import numpy as np
import faiss

def ivf_search(index, Q: np.ndarray, k: int, nprobe: int):
    """
    Budgeted IVF search via nprobe (≥ k) with GPU memory fallback.
    Handles out-of-memory errors by falling back to batch processing.
    
    Args:
        index: FAISS IVF index
        Q: Query vectors [N, D] (L2-normalized for IP)
        k: Number of results to return
        nprobe: Number of clusters to probe (budget)
    
    Returns:
        tuple: (distances, indices, search_time)
    """
    if not np.allclose(np.linalg.norm(Q, axis=1), 1.0, atol=1e-4):
        raise ValueError("Q must be L2-normalized for IP.")
    
    nprobe = int(max(k, nprobe))
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    
    t0 = time.perf_counter()
    try:
        # Try direct search first
        sims, ids = index.search(Q.astype(np.float32), k)
        dt = time.perf_counter() - t0
        return sims, ids, dt
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "alloc fail" in str(e).lower():
            print(f"    Warning: GPU memory insufficient for IVF search, using batch processing")
            # Fallback: process in smaller batches
            batch_size = max(1, len(Q) // 4)  # Process in 4 batches
            ids_list = []
            sims_list = []
            
            for i in range(0, len(Q), batch_size):
                batch_Q = Q[i:i+batch_size]
                batch_sims, batch_ids = index.search(batch_Q.astype(np.float32), k)
                ids_list.append(batch_ids)
                sims_list.append(batch_sims)
            
            ids = np.vstack(ids_list)
            sims = np.vstack(sims_list)
            dt = time.perf_counter() - t0
            return sims, ids, dt
        else:
            raise

# -------------------------
# MAIN TEST (runnable)
# -------------------------
if __name__ == "__main__":
    np.random.seed(0)
    N, d, k = 1_000_000, 128, 10

    # unit-normalized base and queries (so IP ≈ cosine)
    X = np.random.randn(N, d).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Q = np.random.randn(8, d).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12

    # Build IVF index
    print(f"Building IVF index on {N}x{d}...")
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quant, d, 1024, faiss.METRIC_INNER_PRODUCT)
    index.train(X)
    index.add(X)

    # Test search with different nprobe values
    for nprobe in [64, 128, 256, 512]:
        sims, ids, t = ivf_search(index, Q, k=k, nprobe=nprobe)
        print(f"nprobe={nprobe:>4}: time={t:.4f}s, shape={ids.shape}")
        print("  top-5 ids (q0):", ids[0, :5].tolist())
