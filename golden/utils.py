"""
Golden data export utilities for RTL verification.

Provides save/load/compare functions for hex files compatible with $readmemh.
Automatically selects FP16 or FP32 format based on tensor dtype.
"""

import os
import numpy as np
import torch

GOLDEN_ROOT = r"C:\Projects\verilog-test\golden"


def save(dir: str, name: str, t: torch.Tensor):
    """Save tensor to hex file. Auto-selects FP16/FP32 based on dtype."""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float32:
        raw = t.numpy().view(np.uint32).flatten()
        lines = [f"{v:08x}" for v in raw]
    else:
        raw = t.to(torch.float16).numpy().view(np.uint16).flatten()
        lines = [f"{v:04x}" for v in raw]
    path = os.path.join(dir, f"{name}.hex")
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"  {name}.hex: {len(lines)} values")


def load_fp16(dir: str, name: str) -> np.ndarray:
    path = os.path.join(dir, f'{name}.hex')
    with open(path) as f:
        vals = [int(line.strip(), 16) for line in f if line.strip()]
    return np.array(vals, dtype=np.uint16).view(np.float16).astype(np.float64)


def load_fp32(dir: str, name: str) -> np.ndarray:
    path = os.path.join(dir, f'{name}.hex')
    with open(path) as f:
        vals = [int(line.strip(), 16) for line in f if line.strip()]
    return np.array(vals, dtype=np.uint32).view(np.float32).astype(np.float64)


def compare(name: str, tensor: torch.Tensor):
    """Print tensor stats."""
    arr = tensor.detach().cpu().to(torch.float16).numpy().astype(np.float64).flatten()
    n = len(arr)
    abs_arr = np.abs(arr)
    nz = arr[arr != 0]
    print(f"  {name:30s}: n={n:6d}  range=[{arr.min():+.6f}, {arr.max():+.6f}]  "
          f"|mean|={abs_arr.mean():.6f}  nonzero={len(nz)}/{n}")
    return arr


def compare_two(name: str, a_np: np.ndarray, b_np: np.ndarray, tol=0.05):
    """Compare two numpy arrays element-wise, report errors."""
    diff = np.abs(a_np - b_np)
    mask = b_np != 0
    rel_err = np.zeros_like(diff)
    rel_err[mask] = diff[mask] / np.abs(b_np[mask])
    rel_err[~mask & (a_np != 0)] = 1.0

    max_rel = rel_err.max()
    max_abs = diff.max()
    n_exceed = np.sum(rel_err > tol)

    status = "PASS" if n_exceed == 0 else "FAIL"
    print(f"  {status} {name:30s}: max_rel={max_rel*100:.4f}%  max_abs={max_abs:.6f}  "
          f"exceed_{tol*100:.0f}%={n_exceed}/{len(a_np)}")

    if n_exceed > 0 and n_exceed <= 10:
        idxs = np.where(rel_err > tol)[0]
        for idx in idxs[:5]:
            print(f"       idx={idx}: a={a_np[idx]:.6f} b={b_np[idx]:.6f} rel={rel_err[idx]*100:.2f}%")

    return max_rel, n_exceed


def load_model(model_path=r"D:\Development\models\rwkv\rwkv7-g1d-0.1b-20260129-ctx8192"):
    """Load RWKV7 model, return (model, z)."""
    import types
    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.MODEL_NAME = model_path

    from reference.rwkv7 import RWKV_x070
    import reference.rwkv7 as rwkv7_mod
    rwkv7_mod.enable_print = False

    model = RWKV_x070(args)
    return model, model.z
