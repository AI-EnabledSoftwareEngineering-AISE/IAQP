#!/usr/bin/env python3
"""
Download helper for the datasets used by IAQP reproduction.

Supported datasets:
- t2i-10M
- laion-10M

The script follows the existing dataset preparation logic used in the project:
- T2I downloads the first 10M training queries and first 10k public queries via HTTP range requests,
  then patches the FBIN header counts to match the truncated files.
- LAION downloads the required embedding shards, exports them into FBIN files, and downloads the
  10k public query/ground-truth files used by the evaluation pipeline.
"""

from __future__ import annotations

import argparse
import ssl
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np


T2I_BASE_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/"
T2I_GT_URL = "https://zenodo.org/records/11090378/files/t2i.gt.10k.ibin"

LAION_BASE_URL = "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings"
LAION_QUERY_URL = "https://zenodo.org/records/11090378/files/laion.query.10k.fbin"
LAION_GT_URL = "https://zenodo.org/records/11090378/files/laion.gt.10k.ibin"
LAION_PARTS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_file(url: str, dst: Path, range_end: int | None = None) -> None:
    if dst.exists():
        print(f"✓ Found {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇ Downloading {url} -> {dst}")

    req = urllib.request.Request(url)
    if range_end is not None:
        req.add_header("Range", f"bytes=0-{range_end}")

    with urllib.request.urlopen(req, context=_ssl_context()) as response, open(dst, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def patch_fbin_count(path: Path, count: int) -> None:
    with open(path, "r+b") as f:
        f.seek(0)
        f.write(struct.pack("<I", count))
    print(f"✓ Patched FBIN header count: {path} -> {count}")


def write_fbin_from_npy_shards(shards: list[Path], dst: Path) -> None:
    if dst.exists():
        print(f"✓ Found {dst}")
        return

    arrays = [np.load(path, mmap_mode="r") for path in shards]
    total = int(sum(arr.shape[0] for arr in arrays))
    dim = int(arrays[0].shape[1])
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"⇢ Exporting {dst.name} from {len(shards)} shards ({total:,} x {dim})")
    with open(dst, "wb") as f:
        f.write(struct.pack("<II", total, dim))
        for arr in arrays:
            np.asarray(arr, dtype=np.float32).tofile(f)
    print(f"✓ Wrote {dst}")


def download_t2i(root: Path) -> None:
    dataset_dir = root / "t2i-10M"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    need_size = 200 * 4 * 10_000_000 + 8 - 1
    query_10k_size = 200 * 4 * 10_000 + 8 - 1

    download_file(T2I_BASE_URL + "base.10M.fbin", dataset_dir / "base.10M.fbin", range_end=need_size)
    download_file(T2I_BASE_URL + "query.learn.50M.fbin", dataset_dir / "query.train.10M.fbin", range_end=need_size)
    download_file(T2I_BASE_URL + "query.public.100K.fbin", dataset_dir / "query.10k.fbin", range_end=query_10k_size)
    download_file(T2I_GT_URL, dataset_dir / "gt.10k.ibin")

    patch_fbin_count(dataset_dir / "query.train.10M.fbin", 10_000_000)
    patch_fbin_count(dataset_dir / "query.10k.fbin", 10_000)


def download_laion(root: Path) -> None:
    dataset_dir = root / "laion-10M"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    img_shards: list[Path] = []
    txt_shards: list[Path] = []

    for shard_id in LAION_PARTS:
        img_path = dataset_dir / f"img_emb_{shard_id}.npy"
        txt_path = dataset_dir / f"text_emb_{shard_id}.npy"
        download_file(f"{LAION_BASE_URL}/images/{img_path.name}", img_path)
        download_file(f"{LAION_BASE_URL}/texts/{txt_path.name}", txt_path)
        img_shards.append(img_path)
        txt_shards.append(txt_path)

    write_fbin_from_npy_shards(img_shards, dataset_dir / "base.10M.fbin")
    write_fbin_from_npy_shards(txt_shards, dataset_dir / "query.train.10M.fbin")
    download_file(LAION_QUERY_URL, dataset_dir / "query.10k.fbin")
    download_file(LAION_GT_URL, dataset_dir / "gt.10k.ibin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download IAQP datasets.")
    parser.add_argument("dataset", choices=["t2i-10M", "laion-10M"], help="Dataset to download.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Directory where the dataset folder will be created. Default: ./data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "t2i-10M":
        download_t2i(root)
    elif args.dataset == "laion-10M":
        download_laion(root)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"\n✓ Dataset prepared under {root / args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
