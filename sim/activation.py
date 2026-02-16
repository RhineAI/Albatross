"""
RTL-matching activation functions for FP16 tensors.

Replicates the exact behavior of the Verilog LUT-based activation modules:
  - sigmoid_lut.sv  : 64-segment piecewise linear sigmoid
  - tanh_lut.sv     : tanh via 2*sigmoid(2x)-1
  - relu_squared.sv : max(0,x)^2
  - decay_lut.sv    : exp(-exp(-0.5)*sigmoid(x)), fused LUT
  - rsqrt_lut.sv    : 1/sqrt(x), dual-LUT with exponent factoring

All operate on FP16 uint16 representations to match RTL bit-exactly.
Tensor-level wrappers (torch.half -> torch.half) are provided at the bottom.
"""

import numpy as np
import torch
from sim.fp16_arith import fp16_mul, fp16_add, _unpack

# =========================================================
# Sigmoid LUT (64 segments, x in [-8, 8], width=0.25)
# =========================================================

_SIGMOID_LUT = [
    # (slope, intercept) as FP16 uint16
    (0x0e3d, 0x1aec), (0x1001, 0x1c51), (0x1124, 0x1d61), (0x1299, 0x1eb3),
    (0x143c, 0x202b), (0x156f, 0x212e), (0x16f9, 0x226d), (0x1879, 0x23f7),
    (0x19bc, 0x24ee), (0x1b5a, 0x2617), (0x1cb6, 0x2782), (0x1e08, 0x289f),
    (0x1fb8, 0x29ad), (0x20ef, 0x2af3), (0x224d, 0x2c3e), (0x2404, 0x2d29),
    (0x251c, 0x2e41), (0x267d, 0x2f8a), (0x281a, 0x3085), (0x292c, 0x3163),
    (0x2a7b, 0x325e), (0x2c0a, 0x3376), (0x2cfd, 0x3453), (0x2e1b, 0x34f3),
    (0x2f62, 0x3597), (0x3067, 0x3635), (0x3128, 0x36c6), (0x31eb, 0x3740),
    (0x32a4, 0x379d), (0x3343, 0x37d9), (0x33b7, 0x37f7), (0x33f5, 0x3800),
    (0x33f5, 0x3800), (0x33b7, 0x3805), (0x3343, 0x3814), (0x32a4, 0x3832),
    (0x31eb, 0x3860), (0x3128, 0x389d), (0x3067, 0x38e5), (0x2f62, 0x3935),
    (0x2e1b, 0x3986), (0x2cfd, 0x39d6), (0x2c0a, 0x3a23), (0x2a7b, 0x3a68),
    (0x292c, 0x3aa7), (0x281a, 0x3adf), (0x267d, 0x3b0f), (0x251c, 0x3b38),
    (0x2404, 0x3b5b), (0x224d, 0x3b78), (0x20ef, 0x3b91), (0x1fb8, 0x3ba5),
    (0x1e08, 0x3bb6), (0x1cb6, 0x3bc4), (0x1b5a, 0x3bcf), (0x19bc, 0x3bd9),
    (0x1879, 0x3be0), (0x16f9, 0x3be6), (0x156f, 0x3beb), (0x143c, 0x3bef),
    (0x1299, 0x3bf3), (0x1124, 0x3bf5), (0x1001, 0x3bf7), (0x0e3d, 0x3bf9),
]

FP16_ZERO = 0x0000
FP16_ONE  = 0x3C00
FP16_POS8 = 0x4800  # 8.0


def _seg_index(x: int) -> int:
    """Compute segment index for sigmoid/decay LUT, matching RTL exactly.

    Converts FP16 x to Q11.6 fixed-point, adds 8.0, divides by 0.25.
    """
    sign_x, exp_x, mant_x = _unpack(x)
    is_zero = (exp_x == 0)

    # |x| in Q11.6
    mant_full = (1 << 10) | mant_x  # 11-bit with implicit 1
    if is_zero:
        x_abs_q6 = 0
    elif exp_x >= 19:
        x_abs_q6 = mant_full << (exp_x - 19)
    else:
        rshift = 19 - exp_x
        if rshift > 16:
            x_abs_q6 = 0
        else:
            x_abs_q6 = mant_full >> rshift

    # Signed Q11.6
    x_signed_q6 = -x_abs_q6 if sign_x else x_abs_q6

    # x + 8.0 (8.0 in Q11.6 = 512)
    x_plus_8_q6 = x_signed_q6 + 512

    # seg = floor(x_plus_8_q6 / 16), clamped to [0, 63]
    # In RTL: x_plus_8_q6[17:4] with 18-bit signed
    if x_plus_8_q6 < 0:
        return 0
    seg_raw = x_plus_8_q6 >> 4
    if seg_raw > 63:
        return 63
    return seg_raw


def sigmoid_lut(x: int) -> int:
    """Bit-exact replica of sigmoid_lut.sv. Input/output: FP16 uint16."""
    sign_x = (x >> 15) & 1
    exp_x = (x >> 10) & 0x1F
    is_zero = (exp_x == 0)

    # Saturation check: |x| >= 8.0
    x_ge_8 = (not is_zero) and ((x & 0x7FFF) >= (FP16_POS8 & 0x7FFF))
    sat_low = sign_x and x_ge_8    # x <= -8
    sat_high = (not sign_x) and x_ge_8  # x >= 8

    if sat_low:
        return FP16_ZERO
    if sat_high:
        return FP16_ONE

    seg = _seg_index(x)
    slope, intercept = _SIGMOID_LUT[seg]

    sx = fp16_mul(slope, x)
    return fp16_add(sx, intercept)


# =========================================================
# Tanh LUT: tanh(x) = 2*sigmoid(2x) - 1
# =========================================================

def tanh_lut(x: int) -> int:
    """Bit-exact replica of tanh_lut.sv. Input/output: FP16 uint16."""
    sign_x = (x >> 15) & 1
    exp_x = (x >> 10) & 0x1F
    mant_x = x & 0x3FF
    is_zero = (exp_x == 0)

    # Step 1: 2x by incrementing exponent
    if is_zero:
        two_x = 0x0000
    else:
        exp_2x = exp_x + 1
        if exp_2x >= 31:
            # Saturate to ±MAX (exp=30, mant=0x3FF)
            two_x = (sign_x << 15) | (30 << 10) | 0x3FF
        else:
            two_x = (sign_x << 15) | (exp_2x << 10) | mant_x

    # Step 2: sigmoid(2x)
    sig_2x = sigmoid_lut(two_x)

    # Step 3: 2 * sigmoid(2x) by incrementing exponent
    sign_sig = (sig_2x >> 15) & 1
    exp_sig = (sig_2x >> 10) & 0x1F
    mant_sig = sig_2x & 0x3FF
    is_zero_sig = (exp_sig == 0)

    if is_zero_sig:
        two_sig = 0x0000
    elif exp_sig >= 30:
        two_sig = (sign_sig << 15) | (30 << 10) | 0x3FF
    else:
        two_sig = (sign_sig << 15) | ((exp_sig + 1) << 10) | mant_sig

    # Step 4: 2*sigmoid(2x) - 1.0
    neg_one = (1 << 15) | (FP16_ONE & 0x7FFF)  # -1.0 = 0xBC00
    return fp16_add(two_sig, neg_one)


# =========================================================
# ReLU Squared: max(0, x)^2
# =========================================================

def relu_squared(x: int) -> int:
    """Bit-exact replica of relu_squared.sv. Input/output: FP16 uint16."""
    if x & 0x8000:  # sign bit set -> negative
        return 0
    return fp16_mul(x, x)


# =========================================================
# Decay LUT (64 segments, fused exp(-exp(-0.5)*sigmoid(x)))
# =========================================================

_DECAY_LUT = [
    (0x8b91, 0x3bfc), (0x8cdb, 0x3bfb), (0x8e3c, 0x3bf9), (0x9000, 0x3bf8),
    (0x9122, 0x3bf6), (0x9296, 0x3bf3), (0x9439, 0x3bf0), (0x956b, 0x3bed),
    (0x96f2, 0x3be8), (0x9873, 0x3be3), (0x99b3, 0x3bdc), (0x9b4b, 0x3bd3),
    (0x9ca9, 0x3bc9), (0x9df3, 0x3bbd), (0x9f96, 0x3bae), (0xa0d3, 0x3b9d),
    (0xa220, 0x3b88), (0xa3bf, 0x3b70), (0xa4e1, 0x3b54), (0xa61d, 0x3b34),
    (0xa79c, 0x3b10), (0xa8b3, 0x3ae8), (0xa9bf, 0x3abf), (0xaaf1, 0x3a94),
    (0xac21, 0x3a6a), (0xacd4, 0x3a42), (0xad88, 0x3a21), (0xae30, 0x3a06),
    (0xaebd, 0x39f5), (0xaf20, 0x39eb), (0xaf4f, 0x39e8), (0xaf44, 0x39e8),
    (0xaeff, 0x39e8), (0xae89, 0x39e4), (0xadef, 0x39da), (0xad41, 0x39ca),
    (0xac8b, 0x39b3), (0xabb6, 0x3998), (0xaa6f, 0x3979), (0xa94c, 0x3959),
    (0xa84f, 0x393a), (0xa6f4, 0x391c), (0xa591, 0x3900), (0xa46e, 0x38e7),
    (0xa305, 0x38d1), (0xa18a, 0x38be), (0xa05c, 0x38ae), (0x9ed8, 0x38a0),
    (0x9d5e, 0x3894), (0x9c33, 0x388a), (0x9a91, 0x3882), (0x9921, 0x387b),
    (0x9801, 0x3875), (0x9640, 0x3871), (0x94e0, 0x386d), (0x9399, 0x386a),
    (0x91ec, 0x3867), (0x909e, 0x3865), (0x8f32, 0x3864), (0x8d9b, 0x3862),
    (0x8c5e, 0x3861), (0x8ace, 0x3860), (0x894d, 0x3860), (0x8821, 0x385f),
]

FP16_SAT_HIGH_DECAY = 0x385d  # 0.5454


def decay_lut(x: int) -> int:
    """Bit-exact replica of decay_lut.sv. Input/output: FP16 uint16."""
    sign_x = (x >> 15) & 1
    exp_x = (x >> 10) & 0x1F
    is_zero = (exp_x == 0)

    x_ge_8 = (not is_zero) and ((x & 0x7FFF) >= (FP16_POS8 & 0x7FFF))
    sat_low = sign_x and x_ge_8     # x <= -8 -> 1.0
    sat_high = (not sign_x) and x_ge_8  # x >= 8 -> 0.5454

    if sat_low:
        return FP16_ONE
    if sat_high:
        return FP16_SAT_HIGH_DECAY

    seg = _seg_index(x)
    slope, intercept = _DECAY_LUT[seg]

    sx = fp16_mul(slope, x)
    return fp16_add(sx, intercept)


# =========================================================
# RSQRT LUT: 1/sqrt(x), dual-LUT with exponent factoring
# =========================================================

_RSQRT_LUT_EVEN = [
    0x3bf8, 0x3be8, 0x3bd9, 0x3bca, 0x3bbc, 0x3bad, 0x3b9f, 0x3b92,
    0x3b84, 0x3b77, 0x3b6a, 0x3b5e, 0x3b51, 0x3b45, 0x3b39, 0x3b2e,
    0x3b22, 0x3b17, 0x3b0c, 0x3b01, 0x3af6, 0x3aec, 0x3ae2, 0x3ad8,
    0x3ace, 0x3ac4, 0x3aba, 0x3ab1, 0x3aa8, 0x3a9e, 0x3a95, 0x3a8d,
    0x3a84, 0x3a7b, 0x3a73, 0x3a6b, 0x3a62, 0x3a5a, 0x3a52, 0x3a4a,
    0x3a43, 0x3a3b, 0x3a34, 0x3a2c, 0x3a25, 0x3a1e, 0x3a17, 0x3a10,
    0x3a09, 0x3a02, 0x39fb, 0x39f5, 0x39ee, 0x39e7, 0x39e1, 0x39db,
    0x39d5, 0x39ce, 0x39c8, 0x39c2, 0x39bc, 0x39b7, 0x39b1, 0x39ab,
]

_RSQRT_LUT_ODD = [
    0x39a3, 0x3997, 0x398d, 0x3982, 0x3978, 0x396e, 0x3964, 0x395a,
    0x3951, 0x3947, 0x393e, 0x3935, 0x392d, 0x3924, 0x391c, 0x3913,
    0x390b, 0x3903, 0x38fb, 0x38f4, 0x38ec, 0x38e5, 0x38de, 0x38d7,
    0x38cf, 0x38c9, 0x38c2, 0x38bb, 0x38b5, 0x38ae, 0x38a8, 0x38a2,
    0x389b, 0x3895, 0x388f, 0x3889, 0x3884, 0x387e, 0x3878, 0x3873,
    0x386d, 0x3868, 0x3863, 0x385d, 0x3858, 0x3853, 0x384e, 0x3849,
    0x3844, 0x383f, 0x383b, 0x3836, 0x3831, 0x382d, 0x3828, 0x3824,
    0x381f, 0x381b, 0x3817, 0x3812, 0x380e, 0x380a, 0x3806, 0x3802,
]

MAX_POS_15 = 0x7BFF  # FP16 max positive (15-bit magnitude)


def rsqrt_lut(x: int) -> int:
    """Bit-exact replica of rsqrt_lut.sv. Input: FP16 positive uint16."""
    exp_x = (x >> 10) & 0x1F
    mant_x = x & 0x3FF
    is_zero = (exp_x == 0)

    # Exponent parity: exp_x even -> e_ub odd -> use ODD table
    use_odd = (exp_x & 1) == 0

    # e_ub = exp_x - 15 (signed)
    e_ub = exp_x - 15

    # exp_shift calculation (signed arithmetic)
    if not use_odd:
        # EVEN: e_ub is even, exp_shift = -e_ub/2
        # Arithmetic right shift for signed division
        exp_shift = -(e_ub >> 1) if e_ub >= 0 else -(-(-e_ub) >> 1)
        # More precisely: replicate RTL's arithmetic right shift
        exp_shift = -(e_ub // 2)  # Python // is floor division, works for even numbers
    else:
        # ODD: e_ub is odd, exp_shift = -(e_ub-1)/2
        exp_shift = -((e_ub - 1) // 2)

    # Segment index: upper 6 bits of mantissa
    seg_idx = (mant_x >> 4) & 0x3F

    # LUT lookup
    if use_odd:
        lut_val = _RSQRT_LUT_ODD[seg_idx]
    else:
        lut_val = _RSQRT_LUT_EVEN[seg_idx]

    lut_exp = (lut_val >> 10) & 0x1F
    lut_mant = lut_val & 0x3FF

    # Final exponent (signed)
    final_exp = lut_exp + exp_shift

    # Output selection
    if is_zero:
        return 0x7BFF  # rsqrt(0) -> MAX
    if final_exp >= 31:
        return 0x7BFF  # overflow
    if final_exp <= 0:
        return 0x0000  # underflow
    return (final_exp << 10) | lut_mant


# =========================================================
# Torch tensor wrappers: torch.half -> torch.half
# =========================================================

def _to_uint16(t: torch.Tensor) -> np.ndarray:
    """Convert torch.half tensor to numpy uint16 (bit-level view)."""
    return t.detach().cpu().to(torch.float16).numpy().view(np.uint16)


def _from_uint16(arr: np.ndarray, shape, device) -> torch.Tensor:
    """Convert numpy uint16 back to torch.half tensor."""
    return torch.from_numpy(arr.view(np.float16).copy()).to(device).reshape(shape)


def _apply_scalar_fn(fn, t: torch.Tensor) -> torch.Tensor:
    """Apply a scalar uint16->uint16 function element-wise on a torch.half tensor."""
    shape = t.shape
    device = t.device
    arr = _to_uint16(t).flatten()
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = fn(int(arr[i]))
    return _from_uint16(out, shape, device)


def sigmoid_lut_tensor(t: torch.Tensor) -> torch.Tensor:
    """Apply RTL-matching sigmoid LUT to a FP16 tensor."""
    return _apply_scalar_fn(sigmoid_lut, t)


def tanh_lut_tensor(t: torch.Tensor) -> torch.Tensor:
    """Apply RTL-matching tanh LUT to a FP16 tensor."""
    return _apply_scalar_fn(tanh_lut, t)


def relu_squared_tensor(t: torch.Tensor) -> torch.Tensor:
    """Apply RTL-matching ReLU^2 to a FP16 tensor."""
    return _apply_scalar_fn(relu_squared, t)


def decay_lut_tensor(t: torch.Tensor) -> torch.Tensor:
    """Apply RTL-matching decay LUT to a FP16 tensor."""
    return _apply_scalar_fn(decay_lut, t)


def rsqrt_lut_tensor(t: torch.Tensor) -> torch.Tensor:
    """Apply RTL-matching rsqrt LUT to a FP16 tensor."""
    return _apply_scalar_fn(rsqrt_lut, t)
