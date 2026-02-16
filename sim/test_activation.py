"""
Verification: compare RTL-matching activation functions against PyTorch reference.

Reports max absolute error and number of mismatches for each function,
tested over representative FP16 value ranges.
"""

import numpy as np
import torch
import math
from sim.activation import (
    sigmoid_lut_tensor, tanh_lut_tensor, relu_squared_tensor,
    decay_lut_tensor, rsqrt_lut_tensor,
    sigmoid_lut, tanh_lut, relu_squared, decay_lut, rsqrt_lut,
)

EXP_NEG_HALF = math.exp(-0.5)


def _all_fp16_in_range(lo: float, hi: float) -> torch.Tensor:
    """Generate all FP16 values in [lo, hi]."""
    # Generate all 65536 possible uint16 values, interpret as FP16
    all_u16 = np.arange(0, 65536, dtype=np.uint16)
    all_fp16 = all_u16.view(np.float16).astype(np.float64)
    mask = (all_fp16 >= lo) & (all_fp16 <= hi) & np.isfinite(all_fp16)
    selected = all_u16[mask]
    return torch.from_numpy(selected.view(np.float16).copy()).to(torch.float16)


def _compare(name: str, rtl: torch.Tensor, ref: torch.Tensor, tol: float = 0.01):
    """Compare RTL output vs PyTorch reference, report stats."""
    rtl_f64 = rtl.cpu().to(torch.float64).flatten().numpy()
    ref_f64 = ref.cpu().to(torch.float64).flatten().numpy()

    diff = np.abs(rtl_f64 - ref_f64)
    max_abs = diff.max()
    mean_abs = diff.mean()

    # Relative error where ref != 0
    mask = ref_f64 != 0
    rel_err = np.zeros_like(diff)
    rel_err[mask] = diff[mask] / np.abs(ref_f64[mask])
    max_rel = rel_err.max()

    n_exceed = np.sum(diff > tol)
    n_total = len(diff)

    status = "PASS" if max_abs < tol else "INFO"
    print(f"  {status} {name:20s}: n={n_total:6d}  max_abs={max_abs:.6f}  "
          f"mean_abs={mean_abs:.6f}  max_rel={max_rel*100:.2f}%  exceed_{tol}={n_exceed}/{n_total}")

    if n_exceed > 0 and n_exceed <= 5:
        idxs = np.where(diff > tol)[0]
        for idx in idxs:
            print(f"       idx={idx}: rtl={rtl_f64[idx]:.6f} ref={ref_f64[idx]:.6f} diff={diff[idx]:.6f}")


def test_sigmoid():
    print("\n=== Sigmoid LUT vs torch.sigmoid ===")
    x = _all_fp16_in_range(-10.0, 10.0)
    rtl_out = sigmoid_lut_tensor(x)
    ref_out = torch.sigmoid(x.float()).to(torch.float16)
    _compare("sigmoid", rtl_out, ref_out, tol=0.002)


def test_tanh():
    print("\n=== Tanh LUT vs torch.tanh ===")
    x = _all_fp16_in_range(-10.0, 10.0)
    rtl_out = tanh_lut_tensor(x)
    ref_out = torch.tanh(x.float()).to(torch.float16)
    _compare("tanh", rtl_out, ref_out, tol=0.005)


def test_relu_squared():
    print("\n=== ReLU^2 vs torch.relu()^2 ===")
    x = _all_fp16_in_range(-10.0, 10.0)
    rtl_out = relu_squared_tensor(x)
    ref_out = (torch.relu(x.float()) ** 2).to(torch.float16)
    _compare("relu_squared", rtl_out, ref_out, tol=0.001)


def test_decay():
    print("\n=== Decay LUT vs exp(-exp(-0.5)*sigmoid(x)) ===")
    x = _all_fp16_in_range(-10.0, 10.0)
    rtl_out = decay_lut_tensor(x)
    ref_out = torch.exp(-EXP_NEG_HALF * torch.sigmoid(x.float())).to(torch.float16)
    _compare("decay", rtl_out, ref_out, tol=0.002)


def test_rsqrt():
    print("\n=== RSQRT LUT vs 1/sqrt(x) ===")
    # Only positive normalized FP16 values
    all_u16 = np.arange(0, 65536, dtype=np.uint16)
    all_fp16 = all_u16.view(np.float16).astype(np.float64)
    mask = (all_fp16 > 0) & np.isfinite(all_fp16) & (all_u16 >= 0x0400)  # skip denorms
    selected = all_u16[mask]
    x = torch.from_numpy(selected.view(np.float16).copy()).to(torch.float16)

    rtl_out = rsqrt_lut_tensor(x)
    ref_out = torch.rsqrt(x.float()).to(torch.float16)
    _compare("rsqrt", rtl_out, ref_out, tol=0.01)


def test_specific_values():
    """Test a few known values for sanity."""
    print("\n=== Specific value checks ===")

    # sigmoid(0) should be ~0.5
    sig_0 = sigmoid_lut(0x0000)  # x=0
    sig_0_f = np.array([sig_0], dtype=np.uint16).view(np.float16)[0]
    print(f"  sigmoid(0.0) = {sig_0_f:.4f} (expect ~0.5)")

    # sigmoid(-8) should be 0
    sig_neg8 = sigmoid_lut(0xC800)  # -8.0
    print(f"  sigmoid(-8.0) = 0x{sig_neg8:04x} (expect 0x0000)")

    # sigmoid(8) should be 1
    sig_pos8 = sigmoid_lut(0x4800)  # 8.0
    print(f"  sigmoid(8.0) = 0x{sig_pos8:04x} (expect 0x3C00)")

    # tanh(0) should be ~0
    tanh_0 = tanh_lut(0x0000)
    tanh_0_f = np.array([tanh_0], dtype=np.uint16).view(np.float16)[0]
    print(f"  tanh(0.0) = {tanh_0_f:.4f} (expect ~0.0)")

    # relu_squared(-1) should be 0
    relu_neg = relu_squared(0xBC00)  # -1.0
    print(f"  relu_squared(-1.0) = 0x{relu_neg:04x} (expect 0x0000)")

    # relu_squared(2) should be 4
    relu_pos = relu_squared(0x4000)  # 2.0
    relu_pos_f = np.array([relu_pos], dtype=np.uint16).view(np.float16)[0]
    print(f"  relu_squared(2.0) = {relu_pos_f:.4f} (expect 4.0)")

    # rsqrt(1.0) should be ~1.0
    rsqrt_1 = rsqrt_lut(0x3C00)  # 1.0
    rsqrt_1_f = np.array([rsqrt_1], dtype=np.uint16).view(np.float16)[0]
    print(f"  rsqrt(1.0) = {rsqrt_1_f:.4f} (expect ~1.0)")

    # rsqrt(4.0) should be ~0.5
    rsqrt_4 = rsqrt_lut(0x4400)  # 4.0
    rsqrt_4_f = np.array([rsqrt_4], dtype=np.uint16).view(np.float16)[0]
    print(f"  rsqrt(4.0) = {rsqrt_4_f:.4f} (expect ~0.5)")


if __name__ == '__main__':
    test_specific_values()
    test_sigmoid()
    test_tanh()
    test_relu_squared()
    test_decay()
    test_rsqrt()
    print("\nDone!")
