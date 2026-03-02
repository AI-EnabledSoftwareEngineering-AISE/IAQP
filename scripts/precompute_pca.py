#!/usr/bin/env python3
import argparse, pickle, numpy as np, os, torch, sys
from pathlib import Path

# Add the parent directory to the path to import projector modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse your utils
from utils import PCARSpace, l2n_np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_pkl", required=True, help="laion_cache_up_to_*.pkl or datacomp_cache_up_to_*.pkl")
    ap.add_argument("--out_npz",   required=True, help="path to save PCA (npz)")
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--center",    action="store_true",
                    help="center_for_fit=True (subtract mean during fit)")
    ap.add_argument("--d_keep",    type=int, default=0,
                    help="optional projected dim; 0=keep all")
    ap.add_argument("--chunk",     type=int, default=262144,
                    help="rows per chunk when accumulating Gram")
    args = ap.parse_args()

    # ---- Single GPU setup ----
    use_cuda = (args.device.startswith("cuda") and torch.cuda.is_available())
    if use_cuda:
        dev = torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try: 
            torch.set_float32_matmul_precision("high")
        except Exception: 
            pass
    else:
        dev = torch.device("cpu")
    
    print(f"[precompute_pca] device={dev} center_for_fit={args.center}")

    # ---- Load bank ----
    with open(args.cache_pkl, "rb") as f:
        data = pickle.load(f)
    X = data["train"]["image_features"].astype(np.float32)
    X = l2n_np(X)  # keep semantics identical to your class
    N, D = X.shape
    print(f"[precompute_pca] Bank: N={N:,} D={D}")

    # ---- Pass 1: mean (if centering) ----
    if args.center:
        print("[precompute_pca] Computing mean...")
        s_loc = torch.zeros(D, dtype=torch.float64, device=dev)
        for i in range(0, N, args.chunk):
            j = min(N, i + args.chunk)
            xb = torch.as_tensor(X[i:j], device=dev)
            s_loc += xb.sum(dim=0, dtype=torch.float64)

        mu_t = (s_loc / N).to(torch.float32)  # [D] on device
    else:
        mu_t = None

    # ---- Pass 2: Gram G = (X - mu)^T (X - mu) ----
    print("[precompute_pca] Computing Gram matrix...")
    G_loc = torch.zeros(D, D, dtype=torch.float64, device=dev)
    for i in range(0, N, args.chunk):
        j = min(N, i + args.chunk)
        xb = torch.as_tensor(X[i:j], device=dev)  # f32 [b, D]
        if mu_t is not None:
            xb = xb - mu_t
        # accumulate in f64 for stability
        G_loc += (xb.t().matmul(xb)).to(torch.float64)

    # Move to CPU f32 for eigendecomp (small D×D)
    G = G_loc.to(torch.float32).cpu()

    # Eigendecomp and save
    print("[precompute_pca] Computing eigendecomposition...")
    evals, evecs = torch.linalg.eigh(G)       # ascending
    idx = torch.argsort(evals, descending=True)
    V = evecs[:, idx]                         # [D, D]

    if args.d_keep and args.d_keep > 0 and args.d_keep < D:
        V = V[:, :args.d_keep]                # [D, d_keep]

    W = V.t().contiguous().numpy().astype(np.float32)   # rows are directions
    mu = (mu_t.detach().cpu().unsqueeze(0).numpy().astype(np.float32)
          if mu_t is not None else None)

    # Set into your PCARSpace (to "use the class" as requested)
    R = PCARSpace(d_keep=W.shape[0], center_for_fit=args.center, device=str(dev))
    R.W = W
    R.mu = mu

    # Save
    Path(os.path.dirname(args.out_npz) or ".").mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz,
             W=R.W.astype(np.float32),
             mu=(R.mu.astype(np.float32) if R.mu is not None else None),
             center_for_fit=np.array([bool(R.center_for_fit)], dtype=np.bool_))
    print(f"[precompute_pca] ✓ Saved PCA to {args.out_npz}")
    print(f"  W={R.W.shape}, mu={'None' if R.mu is None else R.mu.shape}")

if __name__ == "__main__":
    main()
