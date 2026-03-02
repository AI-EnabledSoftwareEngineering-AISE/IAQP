#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_pca_dims.py
-------------------

Utility script to study PCA dimension choices for LAION and DataComp caches.

Loads a (possibly sampled) subset of normalized CLIP features, fits a GPU-accelerated
PCA (same PCARSpace implementation as cache builders), and reports:
  * cumulative explained variance at each requested dimension
  * residual energy (1 - explained variance)
  * average cosine retention for images and texts after projecting/reconstructing

This helps pick an informed PCA dimension instead of relying on the current
default of 200.
"""
import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add repository root (two levels up) for shared modules
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projector.utils import PCARSpace  # noqa: E402

# Reuse loaders from cache builders
from projector.encode_data import laion10m_cache as laion_cache  # noqa: E402
from projector.encode_data import datacomp_cache as datacomp_cache  # noqa: E402


def parse_int_list(csv: str) -> List[int]:
    values = sorted({int(x.strip()) for x in csv.split(",") if x.strip()})
    if not values:
        raise ValueError("Must provide at least one dimension in --dims")
    return values


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / n).astype(np.float32)


def sample_laion(root: Path, up_to: int, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    parts = [pid for pid in laion_cache.ACCEPTABLE_IDS if pid <= up_to]
    if not parts:
        raise ValueError(f"No LAION parts selected with up_to={up_to}")

    rng = np.random.default_rng(seed)
    data_dir = root / "data"
    imgs, txts = [], []
    total = 0
    target = max_samples if max_samples and max_samples > 0 else None

    for pid in parts:
        img_path = data_dir / f"img_emb_{pid}.npy"
        txt_path = data_dir / f"text_emb_{pid}.npy"
        if not img_path.exists() or not txt_path.exists():
            raise FileNotFoundError(f"Missing LAION shard {pid}: {img_path} / {txt_path}")

        img_part = np.load(str(img_path), mmap_mode="r")
        txt_part = np.load(str(txt_path), mmap_mode="r")
        assert img_part.shape[0] == txt_part.shape[0], f"Shard {pid} mismatch"

        remaining = None if target is None else max(0, target - total)
        take = img_part.shape[0] if remaining is None else min(img_part.shape[0], remaining)
        if take <= 0:
            break

        if take < img_part.shape[0]:
            idx = rng.choice(img_part.shape[0], size=take, replace=False)
            imgs.append(np.array(img_part[idx], dtype=np.float32))
            txts.append(np.array(txt_part[idx], dtype=np.float32))
        else:
            imgs.append(np.array(img_part, dtype=np.float32))
            txts.append(np.array(txt_part, dtype=np.float32))

        total += take
        if target is not None and total >= target:
            break

    X = np.concatenate(imgs, axis=0)
    T = np.concatenate(txts, axis=0)
    meta = {"n_samples": X.shape[0], "dim": X.shape[1], "parts": parts}
    return l2_normalize(X), l2_normalize(T), meta


def sample_datacomp(root: Path, features_pkl: Path, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if not features_pkl.exists():
        raise FileNotFoundError(f"DataComp features pickle not found: {features_pkl}")

    with open(features_pkl, "rb") as f:
        data = pickle.load(f)

    img = data["b32_img"].astype(np.float32)
    txt = data["b32_txt"].astype(np.float32)
    if "image_exists" in data:
        mask = data["image_exists"].astype(bool)
        img = img[mask]
        txt = txt[mask]

    rng = np.random.default_rng(seed)
    if max_samples and max_samples > 0 and img.shape[0] > max_samples:
        idx = rng.choice(img.shape[0], size=max_samples, replace=False)
        img = img[idx]
        txt = txt[idx]

    meta = {"n_samples": img.shape[0], "dim": img.shape[1]}
    return l2_normalize(img), l2_normalize(txt), meta


def stream_component_variance(X: np.ndarray, R: PCARSpace, device: str, chunk: int) -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    W = torch.from_numpy(R.W.astype(np.float32)).to(dev)
    mu = None
    if R.mu is not None:
        mu = torch.from_numpy(np.squeeze(R.mu, axis=0).astype(np.float32)).to(dev)

    var = torch.zeros(W.shape[0], dtype=torch.float64, device=dev)
    N = X.shape[0]
    chunk = max(1, chunk)

    for start in tqdm(range(0, N, chunk), desc="Component variance", unit="vecs"):
        end = min(N, start + chunk)
        xb = torch.from_numpy(X[start:end]).to(dev)
        if mu is not None:
            xb = xb - mu
        proj = xb @ W.T  # [b, D]
        var += (proj.double().pow(2)).sum(dim=0)

    denom = max(1, N - (1 if R.center_for_fit else 0))
    var = (var / denom).cpu().numpy().astype(np.float64)
    return var


def cosine_retention(X: np.ndarray, R: PCARSpace, dims: List[int], device: str, chunk: int) -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    W = torch.from_numpy(R.W.astype(np.float32)).to(dev)
    mu = None
    if R.mu is not None:
        mu = torch.from_numpy(np.squeeze(R.mu, axis=0).astype(np.float32)).to(dev)

    chunk = max(1, chunk)
    total = 0
    cos_sums = torch.zeros(len(dims), dtype=torch.float64, device=dev)
    W_prefix = [W[:d].contiguous() for d in dims]

    for start in tqdm(range(0, X.shape[0], chunk), desc="Cosine retention", unit="vecs"):
        end = min(X.shape[0], start + chunk)
        xb = torch.from_numpy(X[start:end]).to(dev)
        xb_norm = F.normalize(xb, dim=1)
        xb_centered = xb - mu if mu is not None else xb
        proj = xb_centered @ W.T  # [b, D]

        for idx, (d, Wd) in enumerate(zip(dims, W_prefix)):
            proj_d = proj[:, :d]
            recon = proj_d @ Wd
            if mu is not None:
                recon = recon + mu
            recon = F.normalize(recon, dim=1)
            cos = (xb_norm * recon).sum(dim=1)
            cos_sums[idx] += cos.double().sum()

        total += (end - start)

    return (cos_sums / max(1, total)).cpu().numpy()


def analyze_dataset(args) -> Dict:
    if args.dataset == "laion":
        X, T, meta = sample_laion(Path(args.root), args.up_to, args.max_samples, args.seed)
    else:
        features_pkl = Path(args.features_pkl) if args.features_pkl else Path(args.root) / datacomp_cache.DEF_FEATURES_PKL
        X, T, meta = sample_datacomp(Path(args.root), features_pkl, args.max_samples, args.seed)

    dims = [d for d in args.dims if d <= meta["dim"]]
    if not dims:
        raise ValueError(f"All requested dims exceed feature dim={meta['dim']}")

    print(f"Dataset: {args.dataset}")
    print(f"Samples used: {meta['n_samples']:,}  Dim: {meta['dim']}")
    print(f"Target PCA dims: {dims}")
    print(f"Device: {args.device}  Center: {not args.no_center}")

    R = PCARSpace(d_keep=None, center_for_fit=not args.no_center, device=args.device, chunk=args.pca_chunk)
    R.fit(X)

    var = stream_component_variance(X, R, args.device, args.eval_chunk)
    explained = var / var.sum()
    cumulative = np.cumsum(explained)

    img_cos = cosine_retention(X, R, dims, args.device, args.eval_chunk)
    txt_cos = cosine_retention(T, R, dims, args.device, args.eval_chunk)

    rows = []
    for idx, d in enumerate(dims):
        rows.append({
            "dim": d,
            "explained_variance": float(cumulative[d - 1]),
            "residual_variance": float(1.0 - cumulative[d - 1]),
            "image_cosine": float(img_cos[idx]),
            "text_cosine": float(txt_cos[idx]),
        })

    print("\nDim  | Explained% | Residual% | Img Cos | Txt Cos")
    print("-----+------------+-----------+---------+--------")
    for row in rows:
        expl = row["explained_variance"] * 100.0
        resid = row["residual_variance"] * 100.0
        print(f"{row['dim']:4d} | {expl:9.3f}% | {resid:8.3f}% | {row['image_cosine']:.4f} | {row['text_cosine']:.4f}")

    summary = {
        "dataset": args.dataset,
        "root": str(args.root),
        "samples": meta["n_samples"],
        "dim": meta["dim"],
        "centered": not args.no_center,
        "device": args.device,
        "dims": rows,
    }
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Analyze PCA dimension choices for LAION/DataComp.")
    p.add_argument("--dataset", choices=["laion", "datacomp"], required=True, help="Dataset to analyze.")
    p.add_argument("--root", type=str, required=True, help="Dataset root (same as cache builders).")
    p.add_argument("--up_to", type=int, default=10, help="LAION: highest shard id to include (default: 10).")
    p.add_argument("--features_pkl", type=str, default=None, help="DataComp features pickle path (defaults to DEF_FEATURES_PKL under root).")
    p.add_argument("--max_samples", type=int, default=2000000, help="Maximum samples to use for PCA fit (set 0 to use all).")
    p.add_argument("--dims", type=str, default="64,128,160,200,256,320,384,512", help="Comma-separated PCA dims to evaluate.")
    p.add_argument("--device", type=str, default="cuda", help="Device for PCA computations (cuda or cpu).")
    p.add_argument("--pca_chunk", type=int, default=262144, help="Chunk size for PCA covariance accumulation.")
    p.add_argument("--eval_chunk", type=int, default=65536, help="Chunk size for evaluation passes.")
    p.add_argument("--no_center", action="store_true", help="Disable centering before PCA fit.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    p.add_argument("--out", type=str, default=None, help="Optional JSON output file.")
    return p.parse_args()


def main():
    args = parse_args()
    args.dims = parse_int_list(args.dims)
    summary = analyze_dataset(args)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()

