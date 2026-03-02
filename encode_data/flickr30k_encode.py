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
from datasets import load_dataset

import clip


# Hardcoded paths and settings
HF_DATASET_NAME = "nlphuji/flickr30k"
HF_REVISION = "refs/convert/parquet"
HF_CACHE_DIR = "/home/hamed/projects/SPIN/adapter/data/flickr30k_hf"
OUTPUT_DIR = "/home/hamed/projects/SPIN/adapter/data/flickr30k_cache"
MODEL_NAME = "ViT-B/32"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_hf_parquet() -> "datasets.Dataset":
    # Load the one big parquet table; we will create stable split views
    ds = load_dataset(
        HF_DATASET_NAME,
        split="test",
        revision=HF_REVISION,
        cache_dir=HF_CACHE_DIR,
    )
    return ds


def load_flickr30k_data():
    """Load Flickr30K data from HuggingFace dataset"""
    print("Loading Flickr30K dataset from HuggingFace...")
    ds = load_hf_parquet()
    
    print(f"Dataset loaded: {len(ds)} samples")
    print(f"Dataset features: {ds.features}")
    
    # Extract data by split
    splits_data = {}
    for split_name in ['train', 'val', 'test']:
        split_ds = ds.filter(lambda x: x['split'] == split_name)
        print(f"{split_name.upper()} split: {len(split_ds)} images")
        
        image_data = []
        for i, sample in enumerate(split_ds):
            # Images are loaded as PIL objects, no file paths needed
            image_data.append({
                'image': sample['image'],  # PIL Image object
                'captions': sample['caption'],  # List of 5 captions
                'filename': sample['filename'],
                'img_id': sample['img_id'],
                'sample_idx': i
            })
        
        splits_data[split_name] = image_data
        print(f"  {len(image_data)} images, {sum(len(img['captions']) for img in image_data)} captions")
    
    return splits_data


def expand_pairs(image_data):
    """Expand image-caption pairs into individual caption rows"""
    rows = []
    for img_idx, img_data in enumerate(image_data):
        for caption in img_data['captions']:
            rows.append({
                "caption": caption, 
                "img_id": img_idx,
                "filename": img_data['filename']
            })
    return rows


class ImgSet(Dataset):
    def __init__(self, image_data, preprocess):
        self.image_data = image_data
        self.pp = preprocess
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, i):
        # Images are already loaded as PIL objects
        image = self.image_data[i]['image'].convert("RGB")
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
    image_filenames = [img_data['filename'] for img_data in image_data]
    
    pk = {
        "image_features": img.astype(np.float32),
        "text_features": txt.astype(np.float32),
        "text_to_image": txt2img.astype(np.int64),
        "image_filenames": image_filenames,
        "embedding_dim": img.shape[1],
        "model_type": f"openai-clip:{MODEL_NAME}",
    }
    with open(out_dir / f"flickr30k_{name}.pkl", "wb") as f:
        pickle.dump(pk, f)


def brute_force_topk(Q, X, k=10):
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Qn @ Xn.T
    idx = np.argpartition(-S, kth=min(k - 1, X.shape[0] - 1), axis=1)[:, :k]
    srt = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
    return np.take_along_axis(idx, srt, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Encode Flickr30K with CLIP ViT-B/32")
    parser.add_argument("--debug", action="store_true", help="Run on a tiny subset for sanity")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"HF Dataset: {HF_DATASET_NAME}")
    print(f"HF Cache: {HF_CACHE_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    print("Loading CLIP model ...")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    print("Model ready.")

    # Load Flickr30K data
    print("Loading Flickr30K data...")
    splits_data = load_flickr30k_data()
    
    # Process each split separately (like COCO)
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*50}")
        
        image_data = splits_data[split_name]
        
        if args.debug:
            # Keep tiny subset for fast test: 50 images per split
            keep_imgs = 50
            image_data = image_data[:min(keep_imgs, len(image_data))]
            print(f"[DEBUG] Reduced {split_name} size: {len(image_data)} images")

        # Expand ALL caption rows (keep all 5 captions per image for preprocessing)
        print("Expanding ALL caption-image pairs ...")
        rows = expand_pairs(image_data)
        print(f"Total captions: {len(rows)}")

        # Encode images
        print("Encoding images ...")
        img_bs = 256 if not args.debug else 32
        img_feats = encode_images(model, preprocess, device, image_data, batch_size=img_bs, desc=f"{split_name} images")
        print(f"Image shape: {img_feats.shape}")

        # Encode ALL text captions (all 5 per image)
        print("Encoding ALL text captions ...")
        txt_bs = 2048 if not args.debug else 256
        txt_feats, txt2img = encode_text(model, device, rows, batch_size=txt_bs, desc=f"{split_name} text")
        print(f"Text shape: {txt_feats.shape}")

        # Sanity norms
        print("||img|| mean:", float(np.linalg.norm(img_feats, axis=1).mean()))
        print("||txt|| mean:", float(np.linalg.norm(txt_feats, axis=1).mean()))

        # Quick retrieval sanity
        k = 10
        take = min(1000, len(txt_feats))
        if take > 0:
            pred = brute_force_topk(txt_feats[:take], img_feats, k=k)
            hits = (pred == txt2img[:take, None]).any(axis=1).mean()
            print(f"Quick {split_name.upper()}@{k} hit ({take} samples):", round(float(hits), 3))

        # Save cache for this split (with ALL captions)
        print("Saving cache ...")
        dump_split(split_name, img_feats, txt_feats, txt2img, image_data)
        print(f"✅ Wrote {split_name} cache to", OUTPUT_DIR)
    
    print(f"\n{'='*50}")
    print("All splits processed!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
