#!/usr/bin/env python3
# filename: test_faiss_cuvs_ivf_imi.py
import argparse, os, sys, platform, traceback, textwrap
import numpy as np

def np_f32_c_contig(a, name="array"):
    a = np.asarray(a, dtype=np.float32, order="C")
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a, dtype=np.float32)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return a

def banner(t):
    print("\n" + "="*80)
    print(t)
    print("="*80)

def print_array_info(x, name):
    print(f"[{name}] type={type(x)}, dtype={getattr(x, 'dtype', None)}, "
          f"C={getattr(x,'flags',{}).c_contiguous if hasattr(x,'flags') else '?'}, "
          f"shape={getattr(x,'shape', None)}")
    if isinstance(x, np.ndarray):
        print(f"         array_interface? {'__array_interface__' in dir(x)}")

def trycall(label, fn):
    print(f"\n-- {label} --")
    try:
        return True, fn()
    except Exception as e:
        print(f"❌ {label} FAILED: {repr(e)}")
        tb = traceback.format_exc(limit=2)
        for line in tb.rstrip().splitlines():
            print("   " + line)
        return False, None

def dump_env(faiss):
    import importlib
    banner("Environment / Build Info")
    print(f"Python:     {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"OS:         {platform.platform()}")
    print(f"NumPy:      {np.__version__}")
    print(f"FAISS ver:  {getattr(faiss,'__version__','?')}")
    print(f"FAISS file: {getattr(faiss,'__file__','?')}")
    print(f"FAISS GPUs: {faiss.get_num_gpus() if hasattr(faiss,'get_num_gpus') else 'n/a'}")
    # show where cupy/cuvs come from (if present)
    try:
        import cupy, cuvs
        print(f"CuPy:       {cupy.__version__} @ {cupy.__file__}")
        print(f"cuVS:       {cuvs.__version__ if hasattr(cuvs,'__version__') else 'present'}")
    except Exception:
        print("CuPy/cuVS:  not detected")

def make_data(N, D, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D), dtype=np.float32)
    Q = rng.standard_normal((min(1000, max(10, N//50)), D), dtype=np.float32)
    # L2-normalize for IP experiments (optional but common)
    def l2n(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    return l2n(X).astype(np.float32), l2n(Q).astype(np.float32)

def test_ivf_cpu_gpu(faiss, X, Q, D, nlist, k):
    banner("IVF (CPU and GPU if available)")
    X = np_f32_c_contig(X, "X")
    Q = np_f32_c_contig(Q, "Q")
    print_array_info(X, "X")
    print_array_info(Q, "Q")

    # CPU IVF (IP metric; change to L2 if you prefer)
    quant = faiss.IndexFlatIP(D)
    cpu = faiss.IndexIVFFlat(quant, D, int(nlist), faiss.METRIC_INNER_PRODUCT)

    ok, _ = trycall("cpu.train(X)", lambda: cpu.train(X))
    if not ok: return
    ok, _ = trycall("cpu.add(X)",   lambda: cpu.add(X))
    if not ok: return
    ok, out = trycall("cpu.search(Q,k)", lambda: cpu.search(Q, k))
    if ok:
        Dcpu, Icpu = out
        print(f"✅ CPU IVF search OK: D={Dcpu.shape} I={Icpu.shape}")

    # GPU (if supported by this build)
    if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        ok, gpu = trycall("index_cpu_to_gpu", lambda: faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu))
        if not ok:
            print("⚠️ Could not move IVF to GPU (this build may not expose standard GPU shims).")
            return
        ok, out = trycall("gpu.search(Q,k)", lambda: gpu.search(Q, k))
        if ok:
            Dg, Ig = out
            print(f"✅ GPU IVF search OK: D={Dg.shape} I={Ig.shape}")
        else:
            print("⚠️ GPU IVF search failed. If error says 'input not a numpy array', it's likely an ABI/binding mismatch.")

def test_imi_cpu_gpu(faiss, X, Q, D, nbits, k, metric="L2"):
    banner("IMI (CPU; GPU if this FAISS build supports it)")
    X = np_f32_c_contig(X, "X")
    Q = np_f32_c_contig(Q, "Q")
    print_array_info(X, "X")
    print_array_info(Q, "Q")

    # IMI quantizer: MultiIndexQuantizer (2 sub-quantizers, nbits bits each)
    # total nlist ≈ 2^(2*nbits). Keep nbits small (e.g., 8 -> 65536 lists).
    if not hasattr(faiss, "MultiIndexQuantizer"):
        print("❌ This FAISS build has no MultiIndexQuantizer; cannot test IMI.")
        return

    miq = faiss.MultiIndexQuantizer(D, 2, int(nbits))  # m=2
    metric_type = faiss.METRIC_L2 if metric.upper() == "L2" else faiss.METRIC_INNER_PRODUCT
    nlist_est = 1 << (2 * int(nbits))

    # IVF with IMI coarse quantizer, Flat residuals
    index = faiss.IndexIVFFlat(miq, D, nlist_est, metric_type)

    ok, _ = trycall("IMI.train(X)", lambda: index.train(X))
    if not ok: 
        print("⚠️ IMI.train failed (big nlist or unsupported). Try lower --nbits.")
        return
    ok, _ = trycall("IMI.add(X)",   lambda: index.add(X))
    if not ok: return

    ok, out = trycall("IMI.search(Q,k)", lambda: index.search(Q, k))
    if ok:
        Dcpu, Icpu = out
        print(f"✅ IMI CPU search OK: D={Dcpu.shape} I={Icpu.shape}")
    else:
        print("⚠️ IMI CPU search failed; check error details above.")

    # GPU: many FAISS GPU builds don’t support IMI quantizer.
    if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        ok, gpu = trycall("IMI index_cpu_to_gpu", lambda: faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index))
        if not ok:
            print("ℹ️ IMI→GPU not supported in this build (expected for many).")
            return
        ok, out = trycall("IMI GPU search(Q,k)", lambda: gpu.search(Q, k))
        if ok:
            Dg, Ig = out
            print(f"✅ IMI GPU search OK: D={Dg.shape} I={Ig.shape}")
        else:
            print("⚠️ IMI GPU search failed. Likely not supported by this build.")

# Clone original functions from utils.py for testing
def _sha1(x: bytes) -> str:
    import hashlib
    return hashlib.sha1(x).hexdigest()

def resolve_nlist(N: int, hint: int) -> int:
    import math
    return max(16, min(int(4 * math.sqrt(N)), int(hint)))

def bank_fingerprint(X_bank: np.ndarray, R=None) -> str:
    """Stable id for a specific image bank under a specific rotation R."""
    N, D = X_bank.shape
    buf = np.ascontiguousarray(X_bank, dtype=np.float32).tobytes()
    sample = buf[:10**6] + buf[-10**6:] if len(buf) > 2_000_000 else buf
    w_bytes = (np.ascontiguousarray(getattr(R, "W", None), dtype=np.float32).tobytes()
               if getattr(R, "W", None) is not None else b"")
    h = _sha1(sample + w_bytes + str((N, D)).encode())
    return f"N{N}_D{D}_{h[:12]}"

def setup_ivf_gpu_sharding_clone(ivf_cpu_index, world_size: int, local_rank: int, force_cpu: bool = False, faiss_module=None):
    """
    Setup IVF index sharding for GPU training.
    Returns GPU index for current rank or CPU index if force_cpu=True.
    """
    if force_cpu:
        return ivf_cpu_index
    
    if faiss_module is None:
        return ivf_cpu_index
        
    try:
        if hasattr(faiss_module, "get_num_gpus") and faiss_module.get_num_gpus() > 0:
            # Option 1: Create per-GPU IVF (simplest)
            gpu_resource = faiss_module.StandardGpuResources()
            gpu_index = faiss_module.index_cpu_to_gpu(gpu_resource, local_rank, ivf_cpu_index)
            return gpu_index
        else:
            return ivf_cpu_index
    except Exception as e:
        print(f"Warning: Could not move IVF to GPU {local_rank}: {e}")
        return ivf_cpu_index

def get_or_build_ivf_clone(X_R: np.ndarray,
                     indices_dir: str,
                     bank_fp: str,
                     nlist_hint: int,
                     force_cpu: bool = False,
                     num_threads: int = 0,
                     faiss_module=None):
    """
    Returns a FAISS IVFFlat(IP) index (moved to GPU if available), built on
    *rotated, unit-normalized* vectors X_R. CPU copy is cached to disk using a
    rotation-aware fingerprint so different banks/rotations/params don't collide.
    """
    assert faiss_module is not None, "faiss not installed"
    os.makedirs(indices_dir, exist_ok=True)

    nrm = np.linalg.norm(X_R, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-4):
        raise ValueError("get_or_build_ivf: X_R must be L2-normalized.")

    N, D = X_R.shape
    nlist = resolve_nlist(N, nlist_hint)
    print(f"Building IVF index with nlist={nlist}")

    meta = {
        "lib": "faiss",
        "type": "IVFFlat",
        "metric": "IP",
        "nlist": int(nlist),
        "dtype": "f32",
        "N": int(N),
        "D": int(D),
    }
    try:
        meta["faiss_ver"] = getattr(faiss_module, "__version__", "unknown")
    except Exception:
        pass

    fp = _sha1((bank_fp + str(meta)).encode())
    fname = f"ivf_{bank_fp}_{fp[:8]}.faiss"
    meta_name = f"ivf_{bank_fp}_{fp[:8]}.json"
    cpu_path = os.path.join(indices_dir, fname)
    meta_path = os.path.join(indices_dir, meta_name)

    def _build_cpu():
        quant = faiss_module.IndexFlatIP(D)
        cpu = faiss_module.IndexIVFFlat(quant, D, int(nlist), faiss_module.METRIC_INNER_PRODUCT)
        
        # Set number of threads for FAISS
        if num_threads > 0:
            faiss_module.omp_set_num_threads(num_threads)
            print(f"    Using {num_threads} threads for FAISS")
        else:
            import os as os_module
            max_threads = os_module.cpu_count() or 1
            faiss_module.omp_set_num_threads(max_threads)
            print(f"    Using all {max_threads} CPU cores for FAISS")
        
        Xf = X_R.astype(np.float32)
        
        # Use GPU for training if available, fallback to CPU
        try:
            if hasattr(faiss_module, "get_num_gpus") and faiss_module.get_num_gpus() > 0:
                # Create GPU index for training
                gpu_resource = faiss_module.StandardGpuResources()
                gpu_index = faiss_module.index_cpu_to_gpu(gpu_resource, 0, cpu)
                print(f"    Training IVF index on GPU...")
                gpu_index.train(Xf)
                # Move trained index back to CPU for adding
                cpu = faiss_module.index_gpu_to_cpu(gpu_index)
                print(f"    Training completed on GPU, moved back to CPU for adding")
            else:
                print(f"    Training IVF index on CPU (no GPU available)...")
                cpu.train(Xf)
        except Exception as e:
            print(f"    Warning: GPU training failed ({e}), falling back to CPU training")
            cpu.train(Xf)
        
        # Adding always happens on CPU
        print(f"    Adding vectors to IVF index on CPU...")
        cpu.add(Xf)
        return cpu

    # Load or build CPU index
    cpu_idx = None
    if os.path.exists(cpu_path):
        try:
            cpu_idx = faiss_module.read_index(cpu_path)
            # Basic consistency checks
            if cpu_idx.d != D or cpu_idx.ntotal != N:
                cpu_idx = None
                os.remove(cpu_path)
        except Exception:
            try:
                os.remove(cpu_path)
            except Exception:
                pass
            cpu_idx = None

    if cpu_idx is None:
        cpu_idx = _build_cpu()
        try:
            faiss_module.write_index(cpu_idx, cpu_path)
            with open(meta_path, "w") as f:
                import json
                json.dump({**meta, "bank_fp": bank_fp, "file": fname}, f)
        except Exception:
            pass

    # Move to GPU(s) if available, with memory management
    if not force_cpu:
        try:
            if hasattr(faiss_module, "get_num_gpus") and faiss_module.get_num_gpus() > 0:
                # Try to move to GPU, but fallback to CPU if memory issues
                try:
                    gpu_idx = faiss_module.index_cpu_to_all_gpus(cpu_idx)
                    return gpu_idx
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "alloc fail" in str(e).lower():
                        print(f"    Warning: GPU memory insufficient for IVF index, using CPU fallback")
                        return cpu_idx
                    else:
                        raise
        except Exception as e:
            print(f"    Warning: Could not move IVF index to GPU: {e}, using CPU fallback")
    else:
        print(f"    Using CPU evaluation as requested")
    return cpu_idx

def test_trainer_search_pattern(faiss_module, X, Q, D, nlist, k):
    """Test the exact search pattern from trainer.py lines 614-618"""
    banner("Testing Trainer Search Pattern in faiss_cuvs_env")
    X = np_f32_c_contig(X, "X")
    Q = np_f32_c_contig(Q, "Q")
    
    # Simulate the exact trainer.py scenario
    print("\n🔧 Testing trainer.py search pattern...")
    print("  - Building IVF index...")
    
    # Build IVF index (like in trainer.py)
    indices_dir = "/tmp/test_trainer_indices"
    bank_fp = bank_fingerprint(X)
    
    ok, ivf_index = trycall("get_or_build_ivf", lambda: get_or_build_ivf_clone(X, indices_dir, bank_fp, nlist, force_cpu=False, num_threads=0, faiss_module=faiss_module))
    if not ok:
        print("❌ get_or_build_ivf failed")
        return False
    
    # Test the exact trainer.py search pattern
    print("\n🔧 Testing trainer.py search pattern (lines 614-618)...")
    
    # Simulate t_miss (query tensor on GPU)
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_miss = torch.from_numpy(Q).to(device)  # Simulate GPU tensor
    head_k = k
    B = 50  # Budget parameter
    
    print(f"  - t_miss shape: {t_miss.shape}, device: {t_miss.device}")
    print(f"  - head_k: {head_k}, B: {B}")
    
    # Test the exact pattern from trainer.py
    ok, result = trycall("trainer search pattern", lambda: _test_trainer_search_pattern(faiss_module, ivf_index, t_miss, head_k, B, device))
    if ok:
        sims_head_m, ids_head_m = result
        print(f"✅ Trainer search pattern OK:")
        print(f"  - sims_head_m: {sims_head_m.shape}, device: {sims_head_m.device}")
        print(f"  - ids_head_m: {ids_head_m.shape}, device: {ids_head_m.device}")
        return True
    else:
        print("❌ Trainer search pattern failed")
        return False

def _test_trainer_search_pattern(faiss_module, ivf_index, t_miss, head_k, B, device):
    """Exact implementation from trainer.py lines 614-618"""
    import torch
    
    # Line 614: faiss_index = ivf_index
    faiss_index = ivf_index
    
    # Line 615: faiss_index.nprobe = int(B)
    faiss_index.nprobe = int(B)
    
    # Line 616: sims, lab = faiss_index.search(t_miss.detach().cpu().numpy().astype(np.float32), head_k)
    t_miss_np = t_miss.detach().cpu().numpy().astype(np.float32)
    sims, lab = faiss_index.search(t_miss_np, head_k)
    
    # Line 617: ids_head_m = torch.from_numpy(lab.astype(np.int64)).to(device)
    ids_head_m = torch.from_numpy(lab.astype(np.int64)).to(device)
    
    # Line 618: sims_head_m = torch.from_numpy(sims.astype(np.float32)).to(device)
    sims_head_m = torch.from_numpy(sims.astype(np.float32)).to(device)
    
    return sims_head_m, ids_head_m

def test_original_functions(faiss_module, X, Q, D, nlist, k):
    banner("Testing Original SPIN Functions in faiss_cuvs_env")
    X = np_f32_c_contig(X, "X")
    Q = np_f32_c_contig(Q, "Q")
    
    # Test get_or_build_ivf
    print("\n🔧 Testing get_or_build_ivf...")
    indices_dir = "/tmp/test_ivf_indices"
    bank_fp = bank_fingerprint(X)
    
    ok, ivf_index = trycall("get_or_build_ivf", lambda: get_or_build_ivf_clone(X, indices_dir, bank_fp, nlist, force_cpu=False, num_threads=0, faiss_module=faiss_module))
    if not ok:
        print("❌ get_or_build_ivf failed")
        return False
    
    # Test search
    ok, out = trycall("ivf_index.search", lambda: ivf_index.search(Q, k))
    if ok:
        D_result, I_result = out
        print(f"✅ get_or_build_ivf search OK: D={D_result.shape}, I={I_result.shape}")
    else:
        print("❌ get_or_build_ivf search failed")
        return False
    
    # Test setup_ivf_gpu_sharding
    print("\n🔧 Testing setup_ivf_gpu_sharding...")
    world_size = 2
    local_rank = 0
    
    ok, sharded_index = trycall("setup_ivf_gpu_sharding", lambda: setup_ivf_gpu_sharding_clone(ivf_index, world_size, local_rank, force_cpu=False, faiss_module=faiss_module))
    if not ok:
        print("❌ setup_ivf_gpu_sharding failed")
        return False
    
    # Test sharded search
    ok, out = trycall("sharded_index.search", lambda: sharded_index.search(Q, k))
    if ok:
        D_sharded, I_sharded = out
        print(f"✅ setup_ivf_gpu_sharding search OK: D={D_sharded.shape}, I={I_sharded.shape}")
        return True
    else:
        print("❌ setup_ivf_gpu_sharding search failed")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50000, help="db size")
    ap.add_argument("--D", type=int, default=128, help="dim")
    ap.add_argument("--k", type=int, default=10, help="top-k")
    ap.add_argument("--nlist", type=int, default=256, help="IVF lists")
    ap.add_argument("--nbits", type=int, default=8, help="IMI nbits per sub-quantizer")
    args = ap.parse_args()

    banner("Import FAISS")
    try:
        import faiss
    except Exception as e:
        print("❌ import faiss failed:", repr(e))
        sys.exit(1)

    dump_env(faiss)

    # make data
    banner("Generate Data")
    X, Q = make_data(args.N, args.D)
    print_array_info(X, "X (generated)")
    print_array_info(Q, "Q (generated)")

    # sanity: pass something that is NOT numpy to confirm error shape
    def negative_control():
        from array import array
        bad = array("f", [0.0, 1.0, 2.0, 3.0])  # NOT a numpy array
        try:
            faiss.IndexFlatL2(4).add(np_f32_c_contig(bad))  # we coerce so this should still PASS
            print("Negative-control coercion worked (expected).")
        except Exception as e:
            print("Negative-control failed:", repr(e))
    negative_control()

    # Tests
    test_ivf_cpu_gpu(faiss, X, Q, args.D, args.nlist, args.k)
    test_imi_cpu_gpu(faiss, X, Q, args.D, args.nbits, args.k, metric="L2")
    
    # Test original SPIN functions
    test_original_functions(faiss, X, Q, args.D, args.nlist, args.k)
    
    # Test trainer search pattern
    test_trainer_search_pattern(faiss, X, Q, args.D, args.nlist, args.k)

    banner("Done")
    print(textwrap.dedent("""
        If you saw:
        - 'input not a numpy array' during cpu.train/add/search -> your FAISS Python bindings are inconsistent.
        - IVF CPU works but IVF GPU fails with the same message -> this 'faiss-gpu-cuvs' build likely
          doesn’t expose the standard GPU shims your code expects. Keep IVF on CPU in this env, or
          use a separate env with standard FAISS GPU.
        - IMI often works only on CPU; GPU IMI is rarely supported in common builds.
    """).strip())

if __name__ == "__main__":
    main()
