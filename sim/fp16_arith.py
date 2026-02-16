"""
FP16 arithmetic matching RTL (fp16_mul.sv, fp16_add.sv) bit-exactly.

RTL simplifications replicated here:
  - No denorm support (exp=0 treated as zero)
  - No NaN/Inf propagation
  - Truncation (no rounding)
  - Overflow saturates to ±65504 (0x7BFF / 0xFBFF)
  - Underflow flushes to zero

All functions operate on uint16 representations of FP16 values.
"""

import numpy as np

MAX_POS = 0x7BFF  # 65504.0 in FP16


def _unpack(x: int):
    """Unpack FP16 uint16 into (sign, exp, mant)."""
    sign = (x >> 15) & 1
    exp = (x >> 10) & 0x1F
    mant = x & 0x3FF
    return sign, exp, mant


def _pack(sign: int, exp: int, mant: int) -> int:
    return (sign << 15) | (exp << 10) | (mant & 0x3FF)


def fp16_mul(a: int, b: int) -> int:
    """Bit-exact replica of fp16_mul.sv."""
    sign_a, exp_a, mant_a_raw = _unpack(a)
    sign_b, exp_b, mant_b_raw = _unpack(b)

    is_zero_a = (exp_a == 0)
    is_zero_b = (exp_b == 0)

    sign_res = sign_a ^ sign_b

    mant_a = (1 << 10) | mant_a_raw  # 11-bit with implicit 1
    mant_b = (1 << 10) | mant_b_raw

    mant_prod = mant_a * mant_b  # 22-bit

    exp_sum = exp_a + exp_b  # 7-bit range

    # Normalization: two parallel paths
    if mant_prod & (1 << 21):  # mant_prod[21]
        exp_norm = exp_sum - 14  # exp_sum - bias + 1
        mant_norm = (mant_prod >> 11) & 0x3FF
    else:
        exp_norm = exp_sum - 15  # exp_sum - bias
        mant_norm = (mant_prod >> 10) & 0x3FF

    # Output selection (one-hot)
    if is_zero_a or is_zero_b:
        return 0
    if exp_norm < 0:  # underflow (exp_norm[6] set in RTL)
        return 0
    if exp_norm >= 31:  # overflow
        return (sign_res << 15) | MAX_POS
    return (sign_res << 15) | (exp_norm << 10) | mant_norm


def fp16_add(a: int, b: int) -> int:
    """Bit-exact replica of fp16_add.sv (dual-path adder, truncation)."""
    sign_a, exp_a, mant_a_raw = _unpack(a)
    sign_b, exp_b, mant_b_raw = _unpack(b)

    is_zero_a = (exp_a == 0)
    is_zero_b = (exp_b == 0)

    # Implicit bit: normalized=1, zero=0
    mant_a_full = ((0 if is_zero_a else 1) << 10) | mant_a_raw
    mant_b_full = ((0 if is_zero_b else 1) << 10) | mant_b_raw

    # Sort: larger operand first
    if exp_a > exp_b or (exp_a == exp_b and mant_a_full >= mant_b_full):
        a_ge_b = True
    else:
        a_ge_b = False

    if a_ge_b:
        exp_large, mant_large, sign_large = exp_a, mant_a_full, sign_a
        mant_small, sign_small = mant_b_full, sign_b
        exp_diff = exp_a - exp_b
    else:
        exp_large, mant_large, sign_large = exp_b, mant_b_full, sign_b
        mant_small, sign_small = mant_a_full, sign_a
        exp_diff = exp_b - exp_a

    eff_sub = sign_large ^ sign_small
    use_close = eff_sub and (exp_diff <= 1)

    # === FAR path ===
    if exp_diff > 11:
        far_small_aligned = 0
    else:
        far_small_aligned = mant_small >> exp_diff

    if eff_sub:
        far_sum = mant_large - far_small_aligned  # 12-bit range
    else:
        far_sum = mant_large + far_small_aligned

    # FAR normalization
    if far_sum & (1 << 11):  # bit[11] carry
        far_exp_raw = exp_large + 1
        far_mant = (far_sum >> 1) & 0x3FF
    elif far_sum & (1 << 10):  # bit[10] normal
        far_exp_raw = exp_large
        far_mant = far_sum & 0x3FF
    else:  # leading zero from subtraction
        far_exp_raw = exp_large - 1
        far_mant = (far_sum << 1) & 0x3FF

    # === CLOSE path ===
    if exp_diff & 1:
        close_small_aligned = mant_small >> 1
    else:
        close_small_aligned = mant_small

    close_diff = mant_large - close_small_aligned  # 12-bit

    # Leading zero count on 12-bit value
    lzc = 12
    for i in range(12):
        if close_diff & (1 << (11 - i)):
            lzc = i
            break

    close_normalized = (close_diff << lzc) & 0xFFF  # 12-bit
    close_exp_raw = exp_large + 1 - lzc

    close_exp = close_exp_raw & 0x1F
    close_mant = (close_normalized >> 1) & 0x3FF

    # === Merge ===
    sign_res = sign_large
    if use_close:
        add_exp = close_exp
        add_mant = close_mant
        add_exp_full = close_exp_raw
        add_is_zero = (lzc == 12)
    else:
        add_exp = far_exp_raw & 0x1F
        add_mant = far_mant
        add_exp_full = far_exp_raw
        add_is_zero = False

    # === Output selection ===
    if is_zero_a and is_zero_b:
        return 0
    if is_zero_a:
        return b
    if is_zero_b:
        return a

    # Both non-zero: add path
    if add_is_zero:
        return 0
    # Underflow: add_exp_full bit[5] set (negative in 6-bit signed)
    if add_exp_full < 0 or (add_exp_full & 0x20):
        return 0
    if add_exp > 30:
        return (sign_res << 15) | MAX_POS
    return (sign_res << 15) | (add_exp << 10) | add_mant


# =========================================================
# Vectorized wrappers for numpy arrays of uint16
# =========================================================

def fp16_mul_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise fp16_mul on uint16 arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    out = np.empty_like(a_flat)
    for i in range(len(a_flat)):
        out[i] = fp16_mul(int(a_flat[i]), int(b_flat[i]))
    return out.reshape(a.shape)


def fp16_add_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise fp16_add on uint16 arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    out = np.empty_like(a_flat)
    for i in range(len(a_flat)):
        out[i] = fp16_add(int(a_flat[i]), int(b_flat[i]))
    return out.reshape(a.shape)
