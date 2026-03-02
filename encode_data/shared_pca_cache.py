import argparse
from pathlib import Path
from typing import Optional
import numpy as np

from .laion10m_cache import (
    DEF_ROOT as LAION_DEF_ROOT,
    DEF_OUT as LAION_DEF_OUT,
    DEF_UP_TO as LAION_DEF_UP_TO,
    DEF_KEX as LAION_DEF_KEX,
    ACCEPTABLE_IDS as LAION_ACCEPTABLE_IDS,
    ensure_data as ensure_laion_data,
    build_cache as build_laion_cache,
    load_laion_embeddings,
)
from .datacomp_cache import (
    DEF_ROOT as DC_DEF_ROOT,
    DEF_FEATURES_PKL as DC_DEF_FEATURES_PKL,
    DEF_OUT as DC_DEF_OUT,
    DEF_KEX as DC_DEF_KEX,
    DEF_UP_TO as DC_DEF_UP_TO,
    DEF_N_QUERIES as DC_DEF_N_QUERIES,
    build_cache as build_datacomp_cache,
    load_datacomp_embeddings,
)
from ..utils import PCARSpace


def _default_laion_out(root: Path, up_to: int) -> Path:
    precompute_dir = root / "precompute"
    precompute_dir.mkdir(parents=True, exist_ok=True)
    return precompute_dir / LAION_DEF_OUT.format(up_to=up_to)


def _default_datacomp_out(root: Path, up_to: int) -> Path:
    precompute_dir = root / "precompute"
    precompute_dir.mkdir(parents=True, exist_ok=True)
    return precompute_dir / DC_DEF_OUT.format(up_to=up_to)


def _resolve_datacomp_features(root: Path, features_pkl: Optional[str]) -> Path:
    if features_pkl is None:
        return root / DC_DEF_FEATURES_PKL
    path = Path(features_pkl)
    if not path.is_absolute():
        path = root / features_pkl
    return path


def _sample_rows(x: np.ndarray, count: Optional[int], rng: np.random.Generator, label: str) -> np.ndarray:
    if count is None or count <= 0 or count >= x.shape[0]:
        print(f"[{label}] Using all {x.shape[0]:,} rows for PCA fit")
        return x
    idx = rng.choice(x.shape[0], size=count, replace=False)
    print(f"[{label}] Sampling {count:,}/{x.shape[0]:,} rows for PCA fit")
    return x[idx]


def _parse_args():
    p = argparse.ArgumentParser(
        description="Build LAION & DataComp caches that share a single PCA fitted on both datasets."
    )

    # Shared PCA arguments
    p.add_argument("--pca-dim", type=int, required=True, help="Target PCA dimensionality.")
    p.add_argument("--pca-device", type=str, default="cuda", help="Device used for PCA fit/transform.")
    p.add_argument("--pca-center", action="store_true", default=True, help="Center data before PCA (default: True).")
    p.add_argument("--no-pca-center", dest="pca_center", action="store_false", help="Disable centering before PCA.")
    p.add_argument("--pca-laion-count", type=int, default=None, help="Rows from LAION to use for PCA fit (default: all).")
    p.add_argument("--pca-datacomp-count", type=int, default=None, help="Rows from DataComp to use for PCA fit (default: all).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sub-sampling.")

    # LAION-specific arguments
    p.add_argument("--laion-root", type=str, default=LAION_DEF_ROOT, help="LAION dataset root directory.")
    p.add_argument("--laion-up-to", type=int, default=LAION_DEF_UP_TO, choices=LAION_ACCEPTABLE_IDS,
                   help="LAION shard upper bound (inclusive).")
    p.add_argument("--laion-out", type=str, default=None, help="Output pickle filename (under precompute).")
    p.add_argument("--laion-k-exact", type=int, default=LAION_DEF_KEX, help="Exact@K for LAION cache.")
    p.add_argument("--laion-query-batch", type=int, default=4096, help="Query batch size for LAION.")
    p.add_argument("--laion-index-batch", type=int, default=25000, help="Index batch size for LAION.")
    p.add_argument("--laion-use-fp16", action="store_true", help="Enable FP16 matmuls for LAION exact@K.")
    p.add_argument("--laion-force", action="store_true", help="Overwrite existing LAION output.")

    # DataComp-specific arguments
    p.add_argument("--datacomp-root", type=str, default=DC_DEF_ROOT, help="DataComp dataset root directory.")
    p.add_argument("--datacomp-features", type=str, default=None,
                   help="Path to datacomp_features.pkl (defaults to {root}/datacomp_features.pkl).")
    p.add_argument("--datacomp-up-to", type=int, default=DC_DEF_UP_TO, help="#training samples for DataComp.")
    p.add_argument("--datacomp-n-queries", type=int, default=DC_DEF_N_QUERIES, help="#queries for DataComp validation.")
    p.add_argument("--datacomp-out", type=str, default=None, help="Output pickle filename (under precompute).")
    p.add_argument("--datacomp-k-exact", type=int, default=DC_DEF_KEX, help="Exact@K for DataComp cache.")
    p.add_argument("--datacomp-query-batch", type=int, default=4096, help="Query batch size for DataComp.")
    p.add_argument("--datacomp-index-batch", type=int, default=25000, help="Index batch size for DataComp.")
    p.add_argument("--datacomp-use-fp16", action="store_true", help="Enable FP16 matmuls for DataComp exact@K.")
    p.add_argument("--datacomp-force", action="store_true", help="Overwrite existing DataComp output.")

    return p.parse_args()


def main():
    args = _parse_args()

    laion_root = Path(args.laion_root)
    datacomp_root = Path(args.datacomp_root)
    datacomp_features = _resolve_datacomp_features(datacomp_root, args.datacomp_features)

    laion_out = Path(args.laion_out) if args.laion_out else _default_laion_out(laion_root, args.laion_up_to)
    datacomp_out = Path(args.datacomp_out) if args.datacomp_out else _default_datacomp_out(datacomp_root, args.datacomp_up_to)

    if laion_out.exists() and not args.laion_force:
        raise FileExistsError(f"LAION output {laion_out} exists. Use --laion-force to overwrite.")
    if datacomp_out.exists() and not args.datacomp_force:
        raise FileExistsError(f"DataComp output {datacomp_out} exists. Use --datacomp-force to overwrite.")

    print("Ensuring LAION shards are present...")
    ensure_laion_data(laion_root, args.laion_up_to)

    print("Loading LAION embeddings for PCA sampling...")
    laion_imgs, _, _ = load_laion_embeddings(laion_root, args.laion_up_to, normalize=True)
    print("Loading DataComp embeddings for PCA sampling...")
    datacomp_imgs, _, _, _, _ = load_datacomp_embeddings(datacomp_root, datacomp_features, normalize=True)

    rng = np.random.default_rng(args.seed)
    laion_subset = _sample_rows(laion_imgs, args.pca_laion_count, rng, label="LAION")
    datacomp_subset = _sample_rows(datacomp_imgs, args.pca_datacomp_count, rng, label="DataComp")

    concat_inputs = [arr for arr in (laion_subset, datacomp_subset) if arr.size > 0]
    if not concat_inputs:
        raise RuntimeError("No samples provided for PCA fit (both datasets empty?).")
    joint_training = np.concatenate(concat_inputs, axis=0)
    print(f"Fitting shared PCA on {joint_training.shape[0]:,} rows (dim={joint_training.shape[1]})")

    shared_pca = PCARSpace(
        d_keep=args.pca_dim,
        center_for_fit=args.pca_center,
        device=args.pca_device,
        chunk=262144,
    )
    shared_pca.fit(joint_training)
    print(f"✅ Shared PCA trained: W={shared_pca.W.shape}, mu={'None' if shared_pca.mu is None else shared_pca.mu.shape}")

    print("\n=== Building LAION cache with shared PCA ===")
    build_laion_cache(
        root=laion_root,
        out_path=laion_out,
        k_exact=args.laion_k_exact,
        up_to=args.laion_up_to,
        query_batch=args.laion_query_batch,
        index_batch=args.laion_index_batch,
        use_fp16=args.laion_use_fp16,
        pca_dim=None,
        pca_device=args.pca_device,
        pca_center=args.pca_center,
        shared_pca=shared_pca,
    )

    print("\n=== Building DataComp cache with shared PCA ===")
    build_datacomp_cache(
        root=datacomp_root,
        features_pkl=datacomp_features,
        out_path=datacomp_out,
        k_exact=args.datacomp_k_exact,
        up_to=args.datacomp_up_to,
        n_queries=args.datacomp_n_queries,
        query_batch=args.datacomp_query_batch,
        index_batch=args.datacomp_index_batch,
        use_fp16=args.datacomp_use_fp16,
        pca_dim=None,
        pca_device=args.pca_device,
        pca_center=args.pca_center,
        shared_pca=shared_pca,
    )

    print("\n✓ Shared PCA caches written:")
    print(f"   LAION   → {laion_out}")
    print(f"   DataComp→ {datacomp_out}")


if __name__ == "__main__":
    main()

