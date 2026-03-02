#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import clip


# Hardcoded paths and settings
COCO_ROOT = "/ssd/hamed/coco"
ANNOTATIONS_DIR = "/ssd/hamed/coco/annotations"
OUTPUT_DIR = "/home/hamed/projects/SPIN/adapter/data/coco_cache"
MODEL_NAME = "ViT-B/32"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_coco_annotations(split: str):
    """Load COCO annotations for a specific split (train2014 or val2014)"""
    if split not in ["train2014", "val2014"]:
        raise ValueError(f"Split must be 'train2014' or 'val2014', got {split}")
    
    ann_file = Path(ANNOTATIONS_DIR) / f"captions_{split}.json"
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Create image_id to filename mapping
    images = {img['id']: img['file_name'] for img in data['images']}
    
    # Group captions by image_id
    image_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_captions:
            image_captions[img_id] = []
        image_captions[img_id].append(ann['caption'])
    
    # Create list of (image_path, captions) tuples
    image_data = []
    for img_id, filename in images.items():
        if img_id in image_captions:  # Only include images with captions
            image_path = Path(COCO_ROOT) / split / filename
            if image_path.exists():
                image_data.append((image_path, image_captions[img_id]))
    
    return image_data


def expand_pairs(image_data):
    """Expand image-caption pairs into individual caption rows"""
    rows = []
    for img_idx, (image_path, captions) in enumerate(image_data):
        for caption in captions:
            rows.append({
                "caption": caption, 
                "img_id": img_idx,
                "image_path": str(image_path)
            })
    return rows


class ImgSet(Dataset):
    def __init__(self, image_data, preprocess):
        self.image_data = image_data
        self.pp = preprocess
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, i):
        image_path, _ = self.image_data[i]
        image = Image.open(image_path).convert("RGB")
        return self.pp(image), i


class TxtSet(Dataset):
    def __init__(self, rows):
        self.rows = rows
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, i):
        r = self.rows[i]
        return r["caption"], r["img_id"]


@torch.no_grad()
def encode_images(model, preprocess, device, image_data, batch_size=256, num_workers=4, desc="images"):
    ds = ImgSet(image_data, preprocess)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    # ViT-B/32 final projected dim is 512
    embedding_dim = 512
    feats = torch.empty((len(ds), embedding_dim), dtype=torch.float32)

    use_amp = device == "cuda"
    for xb, idx in tqdm(dl, total=len(dl), desc=f"Encoding {desc}"):
        xb = xb.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yb = model.encode_image(xb)
        else:
            yb = model.encode_image(xb)
        yb = torch.nn.functional.normalize(yb.float(), dim=1)
        feats[idx] = yb.cpu()
    return feats.numpy()


@torch.no_grad()
def encode_text(model, device, rows, batch_size=2048, num_workers=2, desc="text"):
    ds = TxtSet(rows)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda batch: list(zip(*batch)),
    )
    embedding_dim = 512
    feats = torch.empty((len(ds), embedding_dim), dtype=torch.float32)
    gt = torch.empty((len(ds),), dtype=torch.long)
    ofs = 0

    use_amp = device == "cuda"
    tokenize = clip.tokenize

    for caps, img_ids in tqdm(dl, total=len(dl), desc=f"Encoding {desc}"):
        toks = tokenize(list(caps), truncate=True).to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yb = model.encode_text(toks)
        else:
            yb = model.encode_text(toks)
        yb = torch.nn.functional.normalize(yb.float(), dim=1)
        n = yb.shape[0]
        feats[ofs : ofs + n] = yb.cpu()
        gt[ofs : ofs + n] = torch.tensor(img_ids, dtype=torch.long)
        ofs += n
    return feats.numpy(), gt.numpy()


def dump_split(name, img, txt, txt2img, image_data):
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filenames from image_data
    image_filenames = [Path(path).name for path, _ in image_data]
    image_paths = [str(path) for path, _ in image_data]
    
    pk = {
        "image_features": img.astype(np.float32),
        "text_features": txt.astype(np.float32),
        "text_to_image": txt2img.astype(np.int64),
        "image_filenames": image_filenames,
        "image_paths": image_paths,
        "embedding_dim": img.shape[1],
        "model_type": f"openai-clip:{MODEL_NAME}",
    }
    with open(out_dir / f"coco_{name}.pkl", "wb") as f:
        pickle.dump(pk, f)


def brute_force_topk(Q, X, k=10):
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Qn @ Xn.T
    idx = np.argpartition(-S, kth=min(k - 1, X.shape[0] - 1), axis=1)[:, :k]
    srt = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
    return np.take_along_axis(idx, srt, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Encode COCO with CLIP ViT-B/32")
    parser.add_argument("--debug", action="store_true", help="Run on a tiny subset for sanity")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"COCO root: {COCO_ROOT}")
    print(f"Annotations: {ANNOTATIONS_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    print("Loading CLIP model ...")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    print("Model ready.")

    # Load COCO data for both splits
    print("Loading COCO annotations ...")
    train_data = load_coco_annotations("train2014")
    val_data = load_coco_annotations("val2014")
    print(f"Train images: {len(train_data)}")
    print(f"Val images: {len(val_data)}")

    if args.debug:
        # Keep tiny subsets for fast test: 10 images each split
        keep_imgs = 10
        train_data = train_data[:min(keep_imgs, len(train_data))]
        val_data = val_data[:min(keep_imgs, len(val_data))]
        print(f"[DEBUG] Reduced sizes: train={len(train_data)}, val={len(val_data)}")

    # Expand caption rows
    print("Expanding caption-image pairs ...")
    train_rows = expand_pairs(train_data)
    val_rows = expand_pairs(val_data)
    print(f"Train captions: {len(train_rows)}")
    print(f"Val captions: {len(val_rows)}")

    # Encode images
    print("Encoding images ...")
    img_bs = 256 if not args.debug else 32
    train_img = encode_images(model, preprocess, device, train_data, batch_size=img_bs, desc="train images")
    val_img = encode_images(model, preprocess, device, val_data, batch_size=img_bs, desc="val images")
    print(f"Image shapes: train={train_img.shape}, val={val_img.shape}")

    # Encode text
    print("Encoding text ...")
    txt_bs = 2048 if not args.debug else 256
    train_txt, train_txt2img = encode_text(model, device, train_rows, batch_size=txt_bs, desc="train text")
    val_txt, val_txt2img = encode_text(model, device, val_rows, batch_size=txt_bs, desc="val text")
    print(f"Text shapes: train={train_txt.shape}, val={val_txt.shape}")

    # Sanity norms
    print("||img|| mean:", float(np.linalg.norm(train_img, axis=1).mean()))
    print("||txt|| mean:", float(np.linalg.norm(train_txt, axis=1).mean()))

    # Quick retrieval sanity on val
    k = 10
    take = min(1000, len(val_txt))
    if take > 0:
        pred = brute_force_topk(val_txt[:take], val_img, k=k)
        hits = (pred == val_txt2img[:take, None]).any(axis=1).mean()
        print(f"Quick Val@{k} hit ({take} samples):", round(float(hits), 3))

    # Save caches
    print("Saving caches ...")
    dump_split("train2014", train_img, train_txt, train_txt2img, train_data)
    dump_split("val2014", val_img, val_txt, val_txt2img, val_data)
    print("✅ Wrote caches to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
