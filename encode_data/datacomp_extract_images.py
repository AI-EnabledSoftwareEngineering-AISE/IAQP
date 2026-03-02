#!/usr/bin/env python3
import os
import io
import sys
import tarfile
import argparse
import time
from pathlib import Path
from typing import List, Tuple

import multiprocessing as mp
from PIL import Image


def _init_worker():
    # Keep PIL fast-safe defaults in each worker
    Image.MAX_IMAGE_PIXELS = None


def find_image_member(by_name, base: str, exts: Tuple[str, ...]) -> tarfile.TarInfo | None:
    for ext in exts:
        cand = base + ext
        m = by_name.get(cand)
        if m is not None and m.isfile():
            return m
    return None


def process_one_shard(shard_path: str, out_dir: str, skip_existing: bool,
                      image_exts: Tuple[str, ...], heartbeat: int = 1000) -> Tuple[str, int, int, int]:
    processed = 0
    skipped = 0
    errors = 0
    shard_name = os.path.basename(shard_path)
    try:
        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()
            by_name = {m.name: m for m in members}
            json_members = [m for m in members if m.isfile() and m.name.endswith(".json")]

            for jm in json_members:
                try:
                    base = jm.name[:-5]
                    im_member = find_image_member(by_name, base, image_exts)
                    if im_member is None:
                        continue
                    jf = tar.extractfile(jm)
                    if jf is None:
                        continue
                    import json as _json
                    try:
                        md = _json.load(jf)
                    except Exception:
                        continue
                    uid = md.get("uid")
                    if not uid:
                        continue
                    # Preserve original encoding/extension
                    ext = os.path.splitext(im_member.name)[1].lower() or ".jpg"
                    out_path = Path(out_dir) / f"{uid}{ext}"
                    if skip_existing and out_path.exists():
                        skipped += 1
                        continue
                    imf = tar.extractfile(im_member)
                    if imf is None:
                        continue
                    # Write original bytes without re-encoding
                    out_tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                    out_tmp.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with open(out_tmp, "wb") as g:
                            g.write(imf.read())
                        os.replace(out_tmp, out_path)
                        processed += 1
                        if heartbeat > 0 and (processed % heartbeat == 0):
                            # best-effort periodic progress from worker
                            print(f"[{shard_name}] processed={processed} skipped={skipped} errors={errors}", flush=True)
                    except Exception:
                        try:
                            if out_tmp.exists():
                                out_tmp.unlink()
                        except Exception:
                            pass
                        errors += 1
                except Exception:
                    errors += 1
                    continue
    except Exception:
        # treat whole shard as error
        return shard_name, processed, skipped, errors + 1

    return shard_name, processed, skipped, errors


def list_shards(shards_dir: Path) -> List[str]:
    return sorted([str(p) for p in shards_dir.glob("*.tar")])


def main():
    ap = argparse.ArgumentParser(description="Extract DataComp shard images preserving original encoding; saved as {uid}{ext}")
    ap.add_argument("--shards_dir", required=True, help="Directory containing .tar shards")
    ap.add_argument("--out_dir", required=True, help="Output directory for extracted images")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2), help="Parallel workers (by shard)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if {uid}.jpg already exists")
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png", help="Candidate image extensions in shards (comma-separated)")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of shards to process (0=all)")
    ap.add_argument("--heartbeat", type=int, default=1000, help="Per-shard progress print every N images (0=disable)")
    args = ap.parse_args()

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = list_shards(shards_dir)
    if args.limit and args.limit > 0:
        shards = shards[: args.limit]
    if not shards:
        print(f"No .tar shards found in {shards_dir}")
        return 1

    image_exts = tuple([s.strip() for s in args.exts.split(',') if s.strip()])
    print(f"Found {len(shards)} shards. Workers={args.workers}. Output={out_dir}")
    t0 = time.time()

    work_items = [(sp, str(out_dir), args.skip_existing, image_exts, int(args.heartbeat)) for sp in shards]

    total_proc = total_skip = total_err = 0
    # Use 'fork' on Linux for faster startup and fewer pickling constraints
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    with ctx.Pool(processes=max(1, int(args.workers)), initializer=_init_worker) as pool:
        for shard_name, proc, skip, err in pool.starmap(process_one_shard, work_items):
            total_proc += proc
            total_skip += skip
            total_err += err
            print(f"[{shard_name}] +{proc} processed, {skip} skipped, {err} errors")

    dt = time.time() - t0
    print("\n=== DONE ===")
    print(f"Processed images: {total_proc}")
    print(f"Skipped existing: {total_skip}")
    print(f"Errors:          {total_err}")
    print(f"Elapsed:         {dt/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())


