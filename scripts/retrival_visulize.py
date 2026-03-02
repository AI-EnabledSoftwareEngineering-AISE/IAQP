#!/usr/bin/env python3
"""
DataComp Retrieval Visualization
Generate HTML galleries comparing baseline vs projected (trained model) retrieval results.

Randomly samples captions from val set and shows top-k retrieved images side-by-side.
"""

import os, sys, time, random
import argparse
import numpy as np
import cupy as cp
import torch
import torch.nn.functional as F
from pathlib import Path
import pickle
import json
import base64
from PIL import Image, ImageDraw, ImageFont

# --- repo imports ---
current_dir = Path(__file__).resolve().parents[2]  # .../t2i_code
sys.path.insert(0, str(current_dir))

from projector.dataset_loader import LaionDatasetLoader
from projector.utils import ResidualProjector, PCARSpace

# --- cuVS CAGRA ---
try:
    from cuvs.neighbors import cagra
except Exception:
    cagra = None

# --- FAISS IVF ---
try:
    import faiss
except Exception:
    faiss = None

# Hard-coded DataComp configuration
DATA_PATH = "/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_3000000.pkl"
IMAGES_DIR = Path("/ssd/hamed/ann/datacomp_small/images")
HIDDEN_DIM = 512
FEATURE_DIM = 512

# Checkpoint paths (CAGRA)
CKPTS_CAGRA = {
    "ep1": "outputs/checkpoints/datacomp-3m_cuvs_cagra_up_to_e3/epoch_1.pt",
    "ep2": "outputs/checkpoints/datacomp-3m_cuvs_cagra_up_to_e3/epoch_2.pt",
    "ep3": "outputs/checkpoints/datacomp-3m_cuvs_cagra_up_to_e3/epoch_3.pt",
}

# Checkpoint paths (IVF)
CKPTS_IVF = {
    "ep1": "outputs/checkpoints/datacomp-3m_ivf_up_to_e3/epoch_1.pt",
    "ep2": "outputs/checkpoints/datacomp-3m_ivf_up_to_e3/epoch_2.pt",
    "ep3": "outputs/checkpoints/datacomp-3m_ivf_up_to_e3/epoch_3.pt",
}

# Defaults
USE_PCA_FROM = "ep2"  # PCA from ep2 for bank space

# CAGRA build config (same as ckpt_shootout_comprehensive)
CAGRA_CFG = dict(
    graph_degree=32,
    intermediate_graph_degree=64,
    nn_descent_niter=30,
    build_algo="nn_descent",
    metric="inner_product",  # rank-equivalent to cosine for L2n inputs
)

# ----------------- helpers -----------------
def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def cpu2gpu(a: np.ndarray) -> cp.ndarray:
    """Convert numpy array to cupy array"""
    a = np.asarray(a, dtype=np.float32, order="C")
    return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32))

def load_ckpt(ckpt_path, device="cuda", dim=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = ck.get("hidden", dim or 512)
    alpha = ck.get("alpha", 0.25)
    model_dim = dim or 512
    model = ResidualProjector(dim=model_dim, hidden=hidden, alpha=alpha)
    msd = ck["model_state_dict"]
    if any(k.startswith("module.") for k in msd):
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
    model.load_state_dict(msd)
    model.to(device).eval()
    R = PCARSpace(d_keep=None, center_for_fit=True, device=device)
    R.W = ck["W"].astype(np.float32)
    R.mu = ck.get("mu", None)
    if R.mu is not None:
        R.mu = R.mu.astype(np.float32)
    return model, R

@torch.no_grad()
def project_np(model, T_np, device="cuda", batch=65536):
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    out = np.empty_like(T_np, dtype=np.float32)
    for s in range(0, T_np.shape[0], batch):
        e = min(T_np.shape[0], s + batch)
        xb = torch.from_numpy(T_np[s:e]).to(dev).float()
        xb = F.normalize(xb, dim=1)
        out[s:e] = model(xb).cpu().numpy().astype(np.float32)
    return out

# ---------- CAGRA ----------
def build_cagra_index(X_gpu):
    """Build CAGRA index on GPU (same as ckpt_shootout_comprehensive)"""
    assert cagra is not None, "cuVS CAGRA not installed"
    p = cagra.IndexParams(
        metric=CAGRA_CFG["metric"],
        intermediate_graph_degree=CAGRA_CFG["intermediate_graph_degree"],
        graph_degree=CAGRA_CFG["graph_degree"],
        build_algo=CAGRA_CFG["build_algo"],
        nn_descent_niter=CAGRA_CFG["nn_descent_niter"],
    )
    t0 = time.time()
    idx = cagra.build(p, X_gpu)
    print(f"✓ CAGRA build done in {time.time()-t0:.2f}s  (gdeg={p.graph_degree}/{p.intermediate_graph_degree}, algo={p.build_algo})")
    return idx

def cagra_search(idx, Q_gpu, k, iters, itopk=256, width=1):
    """CAGRA search with iterations as budget (same as ckpt_shootout_comprehensive)"""
    itopk = max(itopk, k)
    sp = cagra.SearchParams(itopk_size=itopk, search_width=max(1,width), algo="auto", max_iterations=max(1,int(iters)))
    t0 = time.time()
    D, I = cagra.search(sp, idx, Q_gpu, k)
    dt = time.time() - t0
    qps = Q_gpu.shape[0] / max(dt, 1e-9)
    return cp.asnumpy(I), cp.asnumpy(D), qps

# ---------- IVF (FAISS) ----------
import math

def suggested_nlist(N: int) -> int:
    # simple heuristic: 4 * sqrt(N), round to nearest power-of-two-ish bucket
    raw = int(4 * math.sqrt(max(1, N)))
    # snap to {1024, 2048, 4096, 8192}
    if raw <= 1024: return 1024
    if raw <= 2048: return 2048
    if raw <= 4096: return 4096
    return 8192

def build_ivf_index_gpu(X_R_bank: np.ndarray, nlist_hint: int = None):
    assert faiss is not None, "faiss not installed"
    N, D = X_R_bank.shape
    nrm = np.linalg.norm(X_R_bank, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-4):
        raise ValueError("IVF build expects L2-normalized inputs.")
    nlist = nlist_hint if (nlist_hint and nlist_hint>0) else suggested_nlist(N)
    print(f"Building IVF index (IP) with nlist={nlist}")

    quant = faiss.IndexFlatIP(D)
    cpu = faiss.IndexIVFFlat(quant, D, int(nlist), faiss.METRIC_INNER_PRODUCT)

    Xf = X_R_bank.astype(np.float32)
    # train on GPU if possible
    try:
        if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
            print("  • Training IVF on GPU…")
            gpu.train(Xf)
            # move trained back to CPU for add (safer memory-wise), then to GPU for search
            cpu = faiss.index_gpu_to_cpu(gpu)
        else:
            print("  • Training IVF on CPU…")
            cpu.train(Xf)
    except Exception as e:
        print(f"  ! GPU train failed ({e}), fallback to CPU train")
        cpu.train(Xf)

    print("  • Adding vectors to IVF (CPU)…")
    cpu.add(Xf)

    # move to GPU for search if possible
    if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
        print("  • IVF moved to GPU for search")
        return gpu, nlist
    else:
        print("  • Using CPU IVF (no GPU found)")
        return cpu, nlist

def ivf_search(index, Q: np.ndarray, k: int, nprobe: int):
    # clamp nprobe to valid range
    nlist = index.nlist if hasattr(index, "nlist") else nprobe
    nprobe = int(max(1, min(nprobe, nlist)))
    # set nprobe on both CPU and GPU index types
    try:
        index.nprobe = nprobe
    except Exception:
        # some GPU wrappers use ParameterSpace, but most expose nprobe directly
        pass
    t0 = time.time()
    D, I = index.search(Q.astype(np.float32, order="C"), k)
    dt = time.time() - t0
    qps = Q.shape[0] / max(dt, 1e-9)
    return I.astype(np.int64, copy=False), D.astype(np.float32, copy=False), qps, nprobe

# DataComp-specific helpers for visualization
def get_image_path_from_uid(uid: str) -> Path:
    """Get image path from UID (tries multiple extensions)"""
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        candidate = IMAGES_DIR / f"{uid}{ext}"
        if candidate.exists():
            return candidate
    # Return default even if not found (will show error in visualization)
    return IMAGES_DIR / f"{uid}.jpg"

def tokenize_keywords(text: str, max_keywords: int = 5):
    text = text.lower()
    tokens = [t.strip(".,!?;:\"'()[]{}") for t in text.split()]
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "for", "at", "by", "is", "are", "be", "this", "that", "these", "those", "from", "as", "it"}
    kept = [t for t in tokens if t and t not in stop and t.isalpha()]
    seen = set()
    uniq = []
    for t in kept:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= max_keywords:
            break
    return uniq

def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

def image_to_base64(path: str, max_size: int = 200, quality: int = 70) -> str:
    """Load image and convert to base64 for inline embedding"""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        from io import BytesIO
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=quality, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Warning: Could not load image {path}: {e}")
        return ""

def render_gallery_html(output_html: Path, query_captions, gt_indices, bank_paths, baseline_idx, baseline_scores, proj_idx, proj_scores, budget: int = 10, datacomp_anno_map=None):
    output_html.parent.mkdir(parents=True, exist_ok=True)
    
    css = """
    <style>
      body { font-family: Arial, sans-serif; margin: 10px; background: #f5f5f5; max-width: 100%; overflow-x: auto; }
      .container { max-width: 100%; margin: 0 auto; }
      .query { margin: 16px 0; padding: 12px; background: white; border: 1px solid #ddd; border-radius: 6px; overflow-x: auto; }
      .caption { font-weight: bold; margin-bottom: 8px; font-size: 13px; color: #333; }
      .row { display: flex; flex-direction: row; gap: 6px; align-items: flex-start; margin-bottom: 8px; overflow-x: auto; }
      .thumb { position: relative; width: 120px; height: 120px; min-width: 120px; overflow: hidden; border: 2px solid transparent; border-radius: 3px; flex-shrink: 0; }
      .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
      .badge { position: absolute; left: 2px; bottom: 2px; background: rgba(0,0,0,0.8); color: #fff; font-size: 9px; padding: 2px 4px; border-radius: 2px; max-width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .meta { position: absolute; right: 2px; top: 2px; background: rgba(255,255,255,0.95); color: #333; font-size: 9px; padding: 2px 4px; border-radius: 2px; font-weight: bold; }
      .legend { margin: 8px 0 6px 0; font-size: 11px; color: #666; padding: 6px; background: #f0f0f0; border-radius: 3px; }
      .only-proj { border-color: #0a0; }
      .both { border-color: #fa0; }
      .only-base { border-color: #c00; }
      .miss { opacity: 0.7; filter: grayscale(0.4); }
      .rowtitle { width: 70px; font-weight: bold; margin-right: 6px; font-size: 12px; color: #555; padding-top: 50px; flex-shrink: 0; }
      h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 6px; font-size: 18px; margin: 10px 0; }
      .pagination { margin: 20px 0; text-align: center; }
      .pagination button { margin: 0 5px; padding: 8px 16px; font-size: 14px; cursor: pointer; border: 1px solid #ddd; background: white; border-radius: 4px; }
      .pagination button:hover { background: #f0f0f0; }
      .pagination button:disabled { opacity: 0.5; cursor: not-allowed; }
      .pagination-info { margin: 10px 0; font-size: 14px; color: #666; }
      .query.hidden { display: none; }
    </style>
    """
    
    def img_block(path: str, score: float, rank: int, border_class: str, caption: str):
        title = f"#{rank} | {score:.3f}"
        # For DataComp, use keywords from caption
        tags = ",".join(tokenize_keywords(caption, max_keywords=5))
        # Reduce size and quality for large galleries
        img_data = image_to_base64(path, max_size=100, quality=70)
        if not img_data:
            return f'<div class="thumb {border_class}"><div style="width:120px;height:120px;background:#ccc;display:flex;align-items:center;justify-content:center;">Error</div><div class="meta">{html_escape(title)}</div><div class="badge">{html_escape(tags)}</div></div>'
        return f'<div class="thumb {border_class}"><img src="{img_data}" alt="img" loading="lazy"/><div class="meta">{html_escape(title)}</div><div class="badge">{html_escape(tags)}</div></div>'
    
    rows_html = []
    num_queries = len(query_captions)
    budget = min(budget, baseline_idx.shape[1], proj_idx.shape[1])
    
    for qi in range(num_queries):
        caption = query_captions[qi]
        gt = gt_indices[qi] if gt_indices is not None and qi < len(gt_indices) else -1
        base_list = list(baseline_idx[qi][:budget])
        proj_list = list(proj_idx[qi][:budget])
        base_set = set(base_list)
        proj_set = set(proj_list)
        
        # Baseline row
        base_cells = []
        for r, idx in enumerate(base_list, start=1):
            path = bank_paths[idx]
            score = float(baseline_scores[qi][r - 1]) if baseline_scores is not None else 0.0
            border = "both" if idx in proj_set else "only-base"
            base_cells.append(img_block(path, score, r, border, caption))
        
        # Projected row
        proj_cells = []
        for r, idx in enumerate(proj_list, start=1):
            path = bank_paths[idx]
            score = float(proj_scores[qi][r - 1]) if proj_scores is not None else 0.0
            border = "both" if idx in base_set else "only-proj"
            proj_cells.append(img_block(path, score, r, border, caption))
        
        block = f"""
        <div class="query">
          <div class="caption">{html_escape(caption)}</div>
          <div class="legend">Borders: 🟢 green=only projected, 🟠 orange=both, 🔴 red=only baseline.</div>
          <div class="row"><div class="rowtitle">Baseline</div>{''.join(base_cells)}</div>
          <div class="row"><div class="rowtitle">Projected</div>{''.join(proj_cells)}</div>
        </div>
        """
        rows_html.append(block)
    
    # Add pagination JavaScript if we have many queries
    pagination_js = ""
    if num_queries > 10:
        queries_per_page = 10
        total_pages = (num_queries + queries_per_page - 1) // queries_per_page
        pagination_js = f"""
    <script>
      let currentPage = 0;
      const queriesPerPage = {queries_per_page};
      const totalQueries = {num_queries};
      const totalPages = {total_pages};
      
      function showPage(page) {{
        const queries = document.querySelectorAll('.query');
        const startIdx = page * queriesPerPage;
        const endIdx = Math.min(startIdx + queriesPerPage, totalQueries);
        
        queries.forEach((q, idx) => {{
          if (idx >= startIdx && idx < endIdx) {{
            q.classList.remove('hidden');
          }} else {{
            q.classList.add('hidden');
          }}
        }});
        
        document.getElementById('pageInfo').textContent = 
          `Showing queries ${{startIdx + 1}}-${{endIdx}} of ${{totalQueries}} (Page ${{page + 1}} of ${{totalPages}})`;
        
        document.getElementById('prevBtn').disabled = (page === 0);
        document.getElementById('nextBtn').disabled = (page >= totalPages - 1);
      }}
      
      function nextPage() {{
        if (currentPage < totalPages - 1) {{
          currentPage++;
          showPage(currentPage);
        }}
      }}
      
      function prevPage() {{
        if (currentPage > 0) {{
          currentPage--;
          showPage(currentPage);
        }}
      }}
      
      // Initialize
      document.addEventListener('DOMContentLoaded', function() {{
        showPage(0);
      }});
    </script>
    """
        pagination_html = f"""
          <div class="pagination">
            <button id="prevBtn" onclick="prevPage()">Previous</button>
            <span id="pageInfo"></span>
            <button id="nextBtn" onclick="nextPage()">Next</button>
          </div>
        """
    else:
        pagination_html = ""
    
    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DataComp Retrieval Gallery</title>
        {css}
      </head>
      <body>
        <div class="container">
          <h2>DataComp Retrieval Visualization</h2>
          {pagination_html}
          {''.join(rows_html)}
          {pagination_html if pagination_html else ""}
          {pagination_js}
        </div>
      </body>
    </html>
    """
    
    output_html.write_text(html, encoding="utf-8")
    print(f"✅ Wrote gallery to {output_html}")


def render_caption_pdf(out_pdf: Path, caption: str, base_indices, base_scores, proj_indices, proj_scores, bank_paths, num_images: int = 10):
    """Render a single-caption horizontal gallery PDF (Baseline vs Projected) suitable for LaTeX inclusion."""
    num_images = min(num_images, len(base_indices), len(proj_indices))
    # Layout params
    tile_w, tile_h = 240, 240
    gap = 6
    margin = 10
    label_w = 120
    # Compute canvas size
    row_w = label_w + num_images * (tile_w + gap) - gap
    caption_h = 50  # Increased to accommodate caption and legend with proper spacing
    row_h = tile_h
    vgap = 10
    H = margin + caption_h + row_h + vgap + row_h + margin
    W = margin + row_w + margin
    
    canvas = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    # Font
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 22)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_meta = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_meta = ImageFont.load_default()
    
    # Caption text (wrap rudimentarily)
    # Move caption up a bit to avoid overlap with legend
    caption_y = margin + 2
    caption_text = caption
    draw.text((margin, caption_y), caption_text, fill=(20, 20, 20), font=font_title)

    # Legend (colors: green=only projected, blue=both, red=only baseline)
    # Place legend below caption with more spacing to avoid overlap
    legend_y = margin + 28
    BLUE = (0, 120, 255)
    legend_items = [
        ((0, 170, 0), "only projected"),
        (BLUE, "both"),
        ((192, 0, 0), "only baseline"),
    ]
    # Compute legend total width to center it
    def text_w(s):
        try:
            return draw.textlength(s, font=font_meta)
        except Exception:
            return len(s) * 7
    total_w = 0
    for _, label in legend_items:
        total_w += 16 + 6 + text_w(label) + 20  # square + spacing + text + gap
    if total_w > 0:
        total_w -= 20  # remove last gap
    lx = (W - total_w) // 2
    for color, label in legend_items:
        draw.rectangle([lx, legend_y, lx + 16, legend_y + 16], fill=color, outline=(0, 0, 0))
        draw.text((lx + 22, legend_y - 2), label, fill=(60, 60, 60), font=font_meta)
        lx += 22 + text_w(label) + 20
    
    # Rows Y positions
    y0 = margin + caption_h
    y1 = y0 + row_h + vgap
    
    # Row labels
    draw.text((margin, y0 + row_h // 2 - 10), "Baseline", fill=(50, 50, 50), font=font_label)
    draw.text((margin, y1 + row_h // 2 - 10), "Projected", fill=(50, 50, 50), font=font_label)
    
    # Sets to determine border color logic
    base_set = set(int(i) for i in base_indices[:num_images])
    proj_set = set(int(i) for i in proj_indices[:num_images])

    # Helper to place tiles with colored borders
    def paste_row(indices, scores, y, is_projected: bool):
        x = margin + label_w
        for i in range(num_images):
            idx = int(indices[i])
            path = bank_paths[idx]
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            except Exception:
                img = Image.new("RGB", (tile_w, tile_h), color=(200, 200, 200))
            canvas.paste(img, (x, y))
            # Determine border color per legend
            if is_projected:
                if idx in base_set:
                    border_color = BLUE          # both
                else:
                    border_color = (0, 170, 0)    # only projected
            else:
                if idx in proj_set:
                    border_color = BLUE          # both
                else:
                    border_color = (192, 0, 0)    # only baseline
            # Draw border (rectangle outline)
            bt = 3  # border thickness
            for t in range(bt):
                draw.rectangle([x - t, y - t, x + tile_w + t, y + tile_h + t], outline=border_color)
            # Meta box
            score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
            meta = f"#{i+1}  {score:.3f}"
            draw.rectangle([x + tile_w - 100, y + 6, x + tile_w - 6, y + 30], fill=(255, 255, 255))
            draw.text((x + tile_w - 95, y + 8), meta, fill=(0, 0, 0), font=font_meta)
            x += tile_w + gap
    
    paste_row(base_indices, base_scores, y0, is_projected=False)
    paste_row(proj_indices, proj_scores, y1, is_projected=True)
    
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_pdf), "PDF", resolution=300.0)
    print(f"✅ Wrote PDF: {out_pdf}")

# Parse arguments
parser = argparse.ArgumentParser(description="DataComp Retrieval Visualization")
parser.add_argument("--epoch", type=str, choices=["1", "2", "3"], default="2",
                   help="Checkpoint epoch to use (default: 2)")
parser.add_argument("--num_captions", type=int, default=5,
                   help="Number of random captions to sample (default: 5)")
parser.add_argument("--backend", type=str, choices=["cagra", "ivf"], default="cagra",
                   help="Backend to use: 'cagra' or 'ivf' (default: cagra)")
parser.add_argument("--budget", type=int, default=10,
                   help="CAGRA search iterations (budget) or IVF nprobe for retrieval (default: 10)")
parser.add_argument("--num_images", type=int, default=10,
                   help="Number of retrieved images to display per caption (default: 10)")
parser.add_argument("--seed", type=int, default=42,
                   help="Random seed for caption sampling (default: 42)")
parser.add_argument("--output", type=str, default="outputs/visualizations/datacomp_gallery.html",
                   help="Output HTML file path")
parser.add_argument("--latex_out_dir", type=str, default="projector/latex_results/figs",
                   help="If set, write per-caption PDF galleries suitable for LaTeX to this directory")
parser.add_argument("--pdf_render", action="store_true",
                   help="Enable per-caption PDF rendering (default: disabled)")
parser.add_argument("--custom_queries", type=str, default=None,
                   help="Path to custom queries pickle file (generated by generate_queries.py). If provided, uses these queries instead of sampling from dataset.")
parser.add_argument("--auto_select_queries", action="store_true",
                   help="Automatically select queries where projected model shows clear improvement over baseline")
parser.add_argument("--select_top_n", type=int, default=None,
                   help="When using --auto_select_queries, select top N queries showing best improvement (default: use all that show improvement)")
parser.add_argument("--improvement_threshold", type=float, default=0.05,
                   help="Minimum score improvement threshold for auto-selection (default: 0.05, meaning 5%% better)")
parser.add_argument("--save_selection_results", type=str, default=None,
                   help="Path to save selection results JSON file (for use with generate_queries --update_from_selection)")
parser.add_argument("--save_image_paths", type=str, default=None,
                   help="Path to save image paths JSON file with query text, baseline and projected image paths for paper selection")
args = parser.parse_args()

# ----------------- query selection by improvement -----------------
def score_queries_by_improvement(baseline_scores, projected_scores, top_k=3):
    """
    Score queries by how much the projected model improves over baseline.
    
    Args:
        baseline_scores: (n_queries, k) array of baseline retrieval scores
        projected_scores: (n_queries, k) array of projected retrieval scores
        top_k: Number of top results to consider for scoring
    
    Returns:
        improvement_scores: (n_queries,) array of improvement scores
        sorted_indices: Indices sorted by improvement (best first)
    """
    n_queries = baseline_scores.shape[0]
    improvement_scores = np.zeros(n_queries)
    
    for i in range(n_queries):
        # Compare top-k average scores
        base_topk = np.mean(baseline_scores[i][:top_k])
        proj_topk = np.mean(projected_scores[i][:top_k])
        
        # Improvement metric: relative improvement
        if base_topk > 0:
            # Relative improvement: (proj - base) / base
            improvement_scores[i] = (proj_topk - base_topk) / (abs(base_topk) + 1e-8)
        else:
            # If baseline is negative/zero, use absolute improvement
            improvement_scores[i] = proj_topk - base_topk
    
    # Sort by improvement (best first)
    sorted_indices = np.argsort(improvement_scores)[::-1]
    
    return improvement_scores, sorted_indices

def select_best_queries(baseline_scores, projected_scores, improvement_threshold=0.05, select_top_n=None, top_k=3):
    """
    Select queries where projected model shows clear improvement.
    
    Args:
        baseline_scores: (n_queries, k) array of baseline retrieval scores
        projected_scores: (n_queries, k) array of projected retrieval scores
        improvement_threshold: Minimum improvement to consider (relative)
        select_top_n: If set, return only top N queries (None = return all above threshold)
        top_k: Number of top results to consider
    
    Returns:
        selected_indices: Indices of selected queries
        improvement_scores: Improvement scores for all queries
    """
    improvement_scores, sorted_indices = score_queries_by_improvement(
        baseline_scores, projected_scores, top_k=top_k
    )
    
    # Filter by threshold
    above_threshold = improvement_scores >= improvement_threshold
    selected = sorted_indices[above_threshold[sorted_indices]]
    
    # If select_top_n is set, take only top N
    if select_top_n is not None and len(selected) > 0:
        selected = selected[:select_top_n]
    
    return selected, improvement_scores

# ----------------- main -----------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print("DATACOMP RETRIEVAL VISUALIZATION")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading DataComp dataset from {DATA_PATH}...")
    ds = LaionDatasetLoader(DATA_PATH)
    X_train_full, T_train_full, _ = ds.get_train_data()
    X_val_full, T_val_full, _, gt_val = ds.get_split_data("val")
    
    # Get UIDs and texts from cache (for image paths and captions)
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    
    # Get UIDs for train images (for building image paths)
    train_uids = data["train"].get("uid", None)
    train_texts = data["train"].get("text", None)
    
    # Build image paths from UIDs
    if train_uids is not None:
        train_paths = [str(get_image_path_from_uid(str(uid))) for uid in train_uids]
    else:
        print("Warning: UIDs not found in cache, cannot build image paths")
        train_paths = []
    
    # Get query texts (captions) from val split
    val_uids = data["val"].get("uid", None)
    val_texts = data["val"].get("text", None)
    
    print(f"Train: X={X_train_full.shape}, T={T_train_full.shape}")
    print(f"Val: X={X_val_full.shape}, T={T_val_full.shape}")
    print(f"Train UIDs: {len(train_uids) if train_uids is not None else 0}")
    print(f"Val texts: {len(val_texts) if val_texts is not None else 0}")

    # Handle custom queries or sample from dataset
    if args.custom_queries:
        # Load custom queries from file
        custom_queries_path = Path(args.custom_queries)
        if not custom_queries_path.exists():
            raise FileNotFoundError(f"Custom queries file not found: {custom_queries_path}")
        
        print(f"\n📝 Loading custom queries from {custom_queries_path}...")
        with open(custom_queries_path, 'rb') as f:
            custom_data = pickle.load(f)
        
        # Extract queries and features
        query_captions = custom_data.get("queries", [])
        T_val_sample = custom_data.get("features", None)
        
        if T_val_sample is None:
            raise ValueError("Custom queries file must contain 'features' field with CLIP-encoded features")
        
        # Verify features are normalized
        if not custom_data.get("normalized", False):
            print("⚠ Warning: Custom queries features are not normalized. Normalizing...")
            T_val_sample = l2n(T_val_sample)
        
        num_captions = len(query_captions)
        # For custom queries, we don't have ground truth indices
        gt_val_sample = np.full(num_captions, -1, dtype=np.int64)
        
        print(f"✓ Loaded {num_captions} custom queries")
        print(f"  Features shape: {T_val_sample.shape}")
        print(f"  Queries:")
        for i, q in enumerate(query_captions, 1):
            print(f"    {i}. {q}")
    else:
        # Sample random captions from dataset
        num_captions = min(args.num_captions, len(T_val_full))
        sample_indices = np.random.choice(len(T_val_full), size=num_captions, replace=False)
        T_val_sample = T_val_full[sample_indices]
        gt_val_sample = gt_val[sample_indices]
        
        # Get corresponding captions (texts) for sampled queries
        if val_texts is not None:
            query_captions = [str(val_texts[i]) for i in sample_indices]
        else:
            query_captions = [f"Query {i+1}" for i in range(num_captions)]
        
        print(f"\n📸 Sampling {num_captions} random captions from val set")

    # Select checkpoint set based on backend
    if args.backend == "cagra":
        CKPTS = CKPTS_CAGRA
        if cagra is None:
            print("[ERROR] cuVS CAGRA not installed. Install cuVS package.")
            exit(1)
    elif args.backend == "ivf":
        CKPTS = CKPTS_IVF
        if faiss is None:
            print("[ERROR] FAISS not installed. Install faiss-cpu or faiss-gpu.")
            exit(1)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Load checkpoints
    print(f"\nLoading checkpoints for {args.backend.upper()} backend...")
    models = {}
    rotations = {}
    for tag, path in CKPTS.items():
        print(f"  • {tag}: {path}")
        if not os.path.exists(path):
            print(f"[ERROR] Checkpoint not found: {path}")
            exit(1)
        models[tag], rotations[tag] = load_ckpt(path, device="cuda", dim=FEATURE_DIM)

    # Use PCA from specified epoch for bank space
    selected_epoch = f"ep{args.epoch}"
    print(f"\nUsing PCA from {selected_epoch} to define the bank/search space")
    R_ref = rotations[selected_epoch]

    # Transform bank
    print("Rotating bank...")
    X_R_bank = R_ref.transform(X_train_full)
    X_R_bank = l2n(X_R_bank)

    # Build index based on backend
    if args.backend == "cagra":
        print("\nMoving bank to GPU & building CAGRA index...")
        X_gpu = cpu2gpu(X_R_bank)
        print(f"GPU pool used={cp.get_default_memory_pool().used_bytes()/1e9:.3f} GB  ptr={hex(X_gpu.data.ptr)}")
        cagra_index = build_cagra_index(X_gpu)
        _KEEPALIVE = [X_gpu]  # keep bank alive for CAGRA index lifetime
        print("✓ CAGRA ready")
    elif args.backend == "ivf":
        print("\nBuilding IVF index...")
        ivf_index, nlist = build_ivf_index_gpu(X_R_bank)
        print(f"✓ IVF ready (nlist={nlist})")

    # Prepare queries: baseline and projected
    print("\nPreparing queries (baseline + projected)...")
    Q_base = R_ref.transform(T_val_sample)
    Q_base = l2n(Q_base)
    variants = {"baseline": Q_base}
    
    # Project with selected epoch model
    Q_proj = project_np(models[selected_epoch], T_val_sample, device="cuda", batch=65536)
    Q_proj_R = R_ref.transform(Q_proj)
    Q_proj_R = l2n(Q_proj_R)
    variants["projected"] = Q_proj_R
    print(f"✓ Prepared {selected_epoch} projection")
    
    # Perform retrieval based on backend
    # Retrieve enough results to select worst for baseline and best for projected
    retrieval_k = max(args.num_images, 10)  # Retrieve at least num_images, minimum 10
    results = {}
    
    if args.backend == "cagra":
        # Move queries to GPU for CAGRA
        Q_gpu = {name: cpu2gpu(Q) for name, Q in variants.items()}
        
        # budget (args.budget) is the iterations parameter for CAGRA search
        print(f"\nPerforming CAGRA retrieval with budget (iterations)={args.budget}, k={retrieval_k} (will display {args.num_images} images: worst for baseline, best for projected)...")
        for name, Q_gpu_variant in Q_gpu.items():
            I_pred, D_scores, qps = cagra_search(cagra_index, Q_gpu_variant, k=retrieval_k, iters=args.budget, itopk=256, width=1)
            results[name] = {"indices": I_pred, "scores": D_scores}
            print(f"✓ Retrieved top-{retrieval_k} for {name} (budget={args.budget} iterations, QPS={qps:,.0f})")
    
    elif args.backend == "ivf":
        # budget (args.budget) is the nprobe parameter for IVF search
        print(f"\nPerforming IVF retrieval with nprobe={args.budget}, k={retrieval_k} (will display {args.num_images} images: worst for baseline, best for projected)...")
        for name, Q_cpu_variant in variants.items():
            I_pred, D_scores, qps, used_nprobe = ivf_search(ivf_index, Q_cpu_variant, k=retrieval_k, nprobe=args.budget)
            results[name] = {"indices": I_pred, "scores": D_scores}
            print(f"✓ Retrieved top-{retrieval_k} for {name} (nprobe={used_nprobe}, QPS={qps:,.0f})")
    
    # Auto-select queries showing improvement (if enabled)
    # Works with both custom queries and sampled dataset queries
    selected_query_indices = None
    if args.auto_select_queries:
        query_source = "custom queries" if args.custom_queries else "sampled queries"
        print(f"\n🔍 Auto-selecting queries showing improvement from {query_source}...")
        if results["baseline"]["scores"] is None or results["projected"]["scores"] is None:
            print("⚠ Warning: Scores not available, cannot auto-select queries. Disabling auto-selection.")
        else:
            # Use raw scores for selection (before filtering to worst/best)
            baseline_raw_scores = results["baseline"]["scores"]
            projected_raw_scores = results["projected"]["scores"]
            
            selected_query_indices, improvement_scores = select_best_queries(
                baseline_raw_scores,
                projected_raw_scores,
                improvement_threshold=args.improvement_threshold,
                select_top_n=args.select_top_n,
                top_k=min(3, retrieval_k)  # Compare top-3 results
            )
            
            if len(selected_query_indices) == 0:
                print(f"⚠ No queries found above improvement threshold {args.improvement_threshold:.3f}")
                print("   Proceeding with all queries...")
                
                # Save selection results even when none selected (to track failures)
                if args.save_selection_results:
                    orig_captions = query_captions
                    orig_count = len(query_captions) if isinstance(query_captions, list) else query_captions.shape[0]
                    
                    selection_results = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "improvement_threshold": args.improvement_threshold,
                        "select_top_n": args.select_top_n,
                        "total_queries": orig_count,
                        "selected_count": 0,
                        "queries": []
                    }
                    
                    # Add all queries with their improvement scores (all not selected)
                    for i in range(orig_count):
                        query_text = orig_captions[i] if isinstance(orig_captions, list) else orig_captions[i]
                        imp_score = float(improvement_scores[i])
                        
                        selection_results["queries"].append({
                            "index": i,
                            "query": query_text,
                            "improvement_score": imp_score,
                            "selected": False
                        })
                    
                    # Save to JSON
                    results_path = Path(args.save_selection_results)
                    results_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(results_path, "w") as f:
                        json.dump(selection_results, f, indent=2)
                    print(f"\n💾 Saved selection results to: {results_path}")
                    print(f"   All queries were below threshold. Use this file with: python -m projector.scripts.generate_queries --update_from_selection {results_path}")
                
                selected_query_indices = None
            else:
                # Store original captions and count before filtering
                orig_captions = query_captions
                orig_count = len(query_captions) if isinstance(query_captions, list) else query_captions.shape[0]
                
                print(f"✓ Selected {len(selected_query_indices)} queries showing improvement (from {orig_count} total {query_source})")
                print(f"   Improvement scores range: {improvement_scores[selected_query_indices].min():.4f} to {improvement_scores[selected_query_indices].max():.4f}")
                
                print(f"   Selected queries:")
                for idx, qidx in enumerate(selected_query_indices, 1):
                    imp = improvement_scores[qidx]
                    orig_caption = orig_captions[qidx] if isinstance(orig_captions, list) else orig_captions[qidx]
                    print(f"     {idx}. Query {qidx+1}: improvement={imp:.4f} - {orig_caption[:80]}...")
                
                # Save selection results if requested
                if args.save_selection_results:
                    selection_results = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "improvement_threshold": args.improvement_threshold,
                        "select_top_n": args.select_top_n,
                        "total_queries": orig_count,
                        "selected_count": len(selected_query_indices),
                        "queries": []
                    }
                    
                    # Add all queries with their improvement scores and selection status
                    for i in range(orig_count):
                        query_text = orig_captions[i] if isinstance(orig_captions, list) else orig_captions[i]
                        imp_score = float(improvement_scores[i])
                        is_selected = i in selected_query_indices
                        
                        selection_results["queries"].append({
                            "index": i,
                            "query": query_text,
                            "improvement_score": imp_score,
                            "selected": is_selected
                        })
                    
                    # Save to JSON
                    results_path = Path(args.save_selection_results)
                    results_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(results_path, "w") as f:
                        json.dump(selection_results, f, indent=2)
                    print(f"\n💾 Saved selection results to: {results_path}")
                    print(f"   Use this file with: python -m projector.scripts.generate_queries --update_from_selection {results_path}")
                
                # Filter all data to selected queries
                if isinstance(query_captions, list):
                    query_captions = [query_captions[i] for i in selected_query_indices]
                else:
                    query_captions = query_captions[selected_query_indices]
                gt_val_sample = gt_val_sample[selected_query_indices]
                T_val_sample = T_val_sample[selected_query_indices]
                
                # Filter results
                results["baseline"]["indices"] = results["baseline"]["indices"][selected_query_indices]
                results["baseline"]["scores"] = results["baseline"]["scores"][selected_query_indices]
                results["projected"]["indices"] = results["projected"]["indices"][selected_query_indices]
                results["projected"]["scores"] = results["projected"]["scores"][selected_query_indices]
    
    # Post-process results: baseline shows worst, projected shows best
    print(f"\nSelecting images for display:")
    print(f"  Baseline: worst {args.num_images} results (lowest scores)")
    print(f"  Projected: best {args.num_images} results (highest scores)")
    
    # For baseline: take the worst num_images (last num_images from sorted results, which have lowest scores)
    baseline_indices_final = []
    baseline_scores_final = []
    for i in range(len(query_captions)):
        # Results are sorted descending (best first), so worst are at the end
        # Take the last num_images
        start_idx = max(0, retrieval_k - args.num_images)
        baseline_indices_final.append(results["baseline"]["indices"][i][start_idx:])
        if results["baseline"]["scores"] is not None:
            baseline_scores_final.append(results["baseline"]["scores"][i][start_idx:])
        else:
            baseline_scores_final.append(None)
    
    # For projected: take the best num_images (first num_images, which have highest scores)
    projected_indices_final = []
    projected_scores_final = []
    for i in range(len(query_captions)):
        projected_indices_final.append(results["projected"]["indices"][i][:args.num_images])
        if results["projected"]["scores"] is not None:
            projected_scores_final.append(results["projected"]["scores"][i][:args.num_images])
        else:
            projected_scores_final.append(None)
    
    # Convert to numpy arrays for compatibility
    baseline_indices_final = np.array(baseline_indices_final)
    projected_indices_final = np.array(projected_indices_final)
    if baseline_scores_final[0] is not None:
        baseline_scores_final = np.array(baseline_scores_final)
    else:
        baseline_scores_final = None
    if projected_scores_final[0] is not None:
        projected_scores_final = np.array(projected_scores_final)
    else:
        projected_scores_final = None
    
    # Captions are already loaded above from val_texts
    print(f"✓ Using {len(query_captions)} query captions from cache")
    
    # Render gallery
    print(f"\nGenerating visualization gallery...")
    output_path = Path(args.output)
    render_gallery_html(
        output_path,
        query_captions,
        gt_val_sample,
        train_paths,
        baseline_indices_final,
        baseline_scores_final,
        projected_indices_final,
        projected_scores_final,
        budget=args.num_images,  # Display num_images
        datacomp_anno_map=None  # Not used for DataComp
    )
    
    # Save image paths if requested
    if args.save_image_paths:
        print(f"\n💾 Saving image paths to {args.save_image_paths}...")
        image_paths_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_images_per_query": args.num_images,
            "backend": args.backend,
            "epoch": args.epoch,
            "budget": args.budget,
            "queries": []
        }
        
        for i in range(len(query_captions)):
            query_text = query_captions[i] if isinstance(query_captions, list) else str(query_captions[i])
            
            # Baseline images (worst results)
            baseline_images = []
            base_indices = baseline_indices_final[i]
            base_scores = baseline_scores_final[i] if baseline_scores_final is not None else None
            for rank in range(len(base_indices)):
                img_idx = int(base_indices[rank])
                img_path = train_paths[img_idx] if img_idx < len(train_paths) else None
                score = float(base_scores[rank]) if base_scores is not None and rank < len(base_scores) else None
                baseline_images.append({
                    "rank": rank + 1,
                    "image_path": str(img_path) if img_path else None,
                    "score": score,
                    "bank_index": img_idx,
                    "selected": 0  # Default: not selected (0), can be manually changed to 1
                })
            
            # Projected images (best results)
            projected_images = []
            proj_indices = projected_indices_final[i]
            proj_scores = projected_scores_final[i] if projected_scores_final is not None else None
            for rank in range(len(proj_indices)):
                img_idx = int(proj_indices[rank])
                img_path = train_paths[img_idx] if img_idx < len(train_paths) else None
                score = float(proj_scores[rank]) if proj_scores is not None and rank < len(proj_scores) else None
                projected_images.append({
                    "rank": rank + 1,
                    "image_path": str(img_path) if img_path else None,
                    "score": score,
                    "bank_index": img_idx,
                    "selected": 0  # Default: not selected (0), can be manually changed to 1
                })
            
            image_paths_data["queries"].append({
                "query_index": i,
                "query_text": query_text,
                "baseline_images": baseline_images,
                "projected_images": projected_images
            })
        
        # Save to JSON
        image_paths_file = Path(args.save_image_paths)
        image_paths_file.parent.mkdir(parents=True, exist_ok=True)
        with open(image_paths_file, "w") as f:
            json.dump(image_paths_data, f, indent=2)
        print(f"✓ Saved image paths for {len(query_captions)} queries to: {image_paths_file}")
        print(f"   Each query has {args.num_images} baseline (worst) and {args.num_images} projected (best) images")
    
    # Optional: per-caption LaTeX-friendly PDFs
    if args.pdf_render and args.latex_out_dir:
        out_dir = Path(args.latex_out_dir)
        print(f"\nGenerating per-caption PDFs in {out_dir} ...")
        for i in range(len(query_captions)):
            # Use the filtered results (worst for baseline, best for projected)
            base_idx_i = baseline_indices_final[i]
            base_s_i = baseline_scores_final[i] if baseline_scores_final is not None else None
            proj_idx_i = projected_indices_final[i]
            proj_s_i = projected_scores_final[i] if projected_scores_final is not None else None
            # safe filename
            fname = f"datacomp_val_q{i+1:04d}.pdf"
            out_pdf = out_dir / fname
            render_caption_pdf(out_pdf, query_captions[i], base_idx_i, base_s_i, proj_idx_i, proj_s_i, train_paths, num_images=args.num_images)
        print("✓ PDF generation complete")
    
    print(f"\n✅ Done! Gallery saved to: {output_path}")
    print(f"\n📌 To view in Jupyter notebook, run:")
    print(f"   from IPython.display import HTML, IFrame")
    print(f"   display(IFrame('{output_path}', width='100%', height=800))")
    print(f"   # OR:")
    print(f"   # with open('{output_path}', 'r') as f:")
    print(f"   #     display(HTML(f.read()))")

if __name__ == "__main__":
    main()
