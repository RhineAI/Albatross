"""
Minimal 1-layer, 1-head RWKV-7 model with hidden_size=64.
All weights randomly generated. Runs single-token inference using reference functions.
"""
import torch
import torch.nn.functional as F
from reference.rwkv7 import (
    RWKV_x070_TMix_one,
    RWKV_x070_CMix_one,
    RWKV7_ONE_OP,
    HEAD_SIZE,
    DTYPE,
    enable_print,
)

torch.manual_seed(42)

C = 64        # hidden_size
H = 1         # num_heads
N = HEAD_SIZE  # 64
D = 32        # low-rank inner dim
VOCAB = 256   # small vocab for demo
N_LAYER = 1

assert C == H * N

# ── helper: random fp16 tensor on cpu ──
def rp(*shape):
    return (torch.randn(*shape) * 0.02).to(dtype=DTYPE)

def rp32(*shape):
    return torch.randn(*shape) * 0.02

# ── build weight dict (same key convention as reference) ──
z = {}

# embedding (already LN-fused in reference __init__)
z['emb.weight'] = rp(VOCAB, C)

# block 0 layer norms
z['blocks.0.ln1.weight'] = torch.ones(C, dtype=DTYPE)
z['blocks.0.ln1.bias']   = torch.zeros(C, dtype=DTYPE)
z['blocks.0.ln2.weight'] = torch.ones(C, dtype=DTYPE)
z['blocks.0.ln2.bias']   = torch.zeros(C, dtype=DTYPE)

# ── TMix weights ──
att = 'blocks.0.att.'

# token-shift lerp factors [C]
for name in ['x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g']:
    z[att + name] = rp(C)

# w (decay) path
z[att + 'w0'] = rp(C)
z[att + 'w1'] = rp(C, D)
z[att + 'w2'] = rp(D, C)

# a (attention gate) path
z[att + 'a0'] = rp(C)
z[att + 'a1'] = rp(C, D)
z[att + 'a2'] = rp(D, C)

# v_first mix path (ignored for layer 0, but need the tensors)
z[att + 'v0'] = rp(C)
z[att + 'v1'] = rp(C, D)
z[att + 'v2'] = rp(D, C)

# gate path
z[att + 'g1'] = rp(C, D)
z[att + 'g2'] = rp(D, C)

# per-channel vectors [C]
z[att + 'k_k'] = rp(C)
z[att + 'k_a'] = rp(C)
z[att + 'r_k'] = rp(C)  # originally [H,N] flattened

# projection matrices [C, C] (reference transposes them at load time)
z[att + 'receptance.weight'] = rp(C, C)
z[att + 'key.weight']        = rp(C, C)
z[att + 'value.weight']      = rp(C, C)
z[att + 'output.weight']     = rp(C, C)

# group norm params [C]
z[att + 'ln_x.weight'] = torch.ones(C, dtype=DTYPE)
z[att + 'ln_x.bias']   = torch.zeros(C, dtype=DTYPE)

# ── CMix weights ──
ffn = 'blocks.0.ffn.'
z[ffn + 'x_k']          = rp(C)
# CMix uses a wider hidden: typically 3.5x or 4x, use 4*C here
CMix_D = C * 4
z[ffn + 'key.weight']   = rp(C, CMix_D)
z[ffn + 'value.weight'] = rp(CMix_D, C)

# ── final output ──
z['ln_out.weight'] = torch.ones(C, dtype=DTYPE)
z['ln_out.bias']   = torch.zeros(C, dtype=DTYPE)
z['head.weight']   = rp(C, VOCAB)

# ══════════════════════════════════════════
#  Single-token inference
# ══════════════════════════════════════════
def forward_one(token_id: int):
    """Run one token through the 1-layer RWKV-7 model, return logits [VOCAB]."""
    # init state
    # state[0]: [n_layer, 2, C]  (token-shift states for TMix & CMix)
    # state[1]: [n_layer, H, N, N] (wkv state, fp32)
    state = [
        torch.zeros(N_LAYER, 2, C, dtype=DTYPE),
        torch.zeros(N_LAYER, H, N, N, dtype=torch.float32),
    ]

    x = z['emb.weight'][token_id]  # [C]
    v_first = torch.empty_like(x)

    i = 0  # single layer
    bbb = f'blocks.{i}.'
    a = f'blocks.{i}.att.'
    f_ = f'blocks.{i}.ffn.'

    # LN1 → TMix
    xx = F.layer_norm(x, (C,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
    xx, v_first = RWKV_x070_TMix_one(
        i, H, N, xx, state[0][i], v_first, state[1][i],
        z[a+'x_r'], z[a+'x_w'], z[a+'x_k'], z[a+'x_v'], z[a+'x_a'], z[a+'x_g'],
        z[a+'w0'], z[a+'w1'], z[a+'w2'],
        z[a+'a0'], z[a+'a1'], z[a+'a2'],
        z[a+'v0'], z[a+'v1'], z[a+'v2'],
        z[a+'g1'], z[a+'g2'],
        z[a+'k_k'], z[a+'k_a'], z[a+'r_k'],
        z[a+'receptance.weight'], z[a+'key.weight'], z[a+'value.weight'], z[a+'output.weight'],
        z[a+'ln_x.weight'], z[a+'ln_x.bias'],
    )
    x = x + xx

    # LN2 → CMix
    xx = F.layer_norm(x, (C,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
    xx = RWKV_x070_CMix_one(xx, state[0][i], z[f_+'x_k'], z[f_+'key.weight'], z[f_+'value.weight'])
    x = x + xx

    # final LN → head
    x = F.layer_norm(x, (C,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
    logits = x @ z['head.weight']
    return logits


if __name__ == '__main__':
    token_id = 42
    logits = forward_one(token_id)
    print(f"\nInput token: {token_id}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Top-5 token ids: {torch.topk(logits.float(), 5).indices.tolist()}")
    print(f"Top-5 logits:    {torch.topk(logits.float(), 5).values.tolist()}")
    print("\nInference OK!")
