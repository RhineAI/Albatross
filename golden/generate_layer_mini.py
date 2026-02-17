"""
RWKV7 Layer mini golden data generator — 整层端到端测试。

数据流: x_in → LN1 → TMix → +residual → LN2 → CMix → +residual → x_out
维度: C=128, H=2, N=64, FFN_DIM=512, D_W/A/V=16, D_G=32

复用 generate_tmix_mini / generate_cmix_mini 的核心计算函数，避免重复代码。

输出: C:/Projects/verilog-test/golden/mini/layer/ 下的 hex 文件
用法: cd C:/Projects/Albatross && uv run -m golden.generate_layer_mini
"""

import os
import glob
import torch
import torch.nn.functional as F
from golden.utils import save
from golden.generate_tmix_mini import RWKV_x070_TMix_one_export
from golden.generate_cmix_mini import RWKV_x070_CMix_one_export
from reference.rwkv7 import DTYPE

C, H, N = 128, 2, 64
FFN_DIM = 512
D_W, D_A, D_V, D_G = 16, 16, 16, 32

OUT_DIR = r"C:\Projects\verilog-test\golden\mini\layer"


def main():
    if os.path.exists(OUT_DIR):
        for f in glob.glob(os.path.join(OUT_DIR, '*.hex')):
            os.remove(f)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(42)
    device = "cuda"

    print(f"Generating layer mini golden data")
    print(f"  C={C}, H={H}, N={N}, FFN_DIM={FFN_DIM}")
    print(f"  D_W={D_W}, D_A={D_A}, D_V={D_V}, D_G={D_G}")
    print(f"  Output: {OUT_DIR}/\n")

    # ---- 生成随机输入 ----
    x_in = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    tmix_x_prev = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    cmix_x_prev = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    v_first_in = torch.zeros(C, device=device, dtype=DTYPE)
    att_state = torch.zeros(H, N, N, device=device, dtype=torch.float32)

    # ---- LN1 权重 ----
    ln1_w = (torch.rand(C, device=device) * 0.5 + 0.75).to(DTYPE)
    ln1_b = (torch.randn(C, device=device) * 0.1).to(DTYPE)

    # ---- TMix 权重 ----
    x_r = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    x_w = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    x_k = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    x_v = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    x_a = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    x_g = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)

    R_ = (torch.randn(C, C, device=device) * 0.1).to(DTYPE)
    K_ = (torch.randn(C, C, device=device) * 0.1).to(DTYPE)
    V_ = (torch.randn(C, C, device=device) * 0.1).to(DTYPE)
    O_ = (torch.randn(C, C, device=device) * 0.1).to(DTYPE)

    w1 = (torch.randn(C, D_W, device=device) * 0.3).to(DTYPE)
    w2 = (torch.randn(D_W, C, device=device) * 0.3).to(DTYPE)
    a1 = (torch.randn(C, D_A, device=device) * 0.3).to(DTYPE)
    a2 = (torch.randn(D_A, C, device=device) * 0.3).to(DTYPE)
    v1 = (torch.randn(C, D_V, device=device) * 0.3).to(DTYPE)
    v2 = (torch.randn(D_V, C, device=device) * 0.3).to(DTYPE)
    g1 = (torch.randn(C, D_G, device=device) * 0.3).to(DTYPE)
    g2 = (torch.randn(D_G, C, device=device) * 0.3).to(DTYPE)

    w0 = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    a0 = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    v0 = (torch.randn(C, device=device) * 0.5).to(DTYPE)

    k_k = (torch.randn(C, device=device) * 0.3).to(DTYPE)
    k_a = (torch.randn(C, device=device) * 0.3).to(DTYPE)
    r_k = (torch.randn(C, device=device) * 0.3).to(DTYPE)

    att_ln_w = torch.ones(C, device=device, dtype=DTYPE)
    att_ln_b = torch.zeros(C, device=device, dtype=DTYPE)

    # ---- LN2 权重 ----
    ln2_w = (torch.rand(C, device=device) * 0.5 + 0.75).to(DTYPE)
    ln2_b = (torch.randn(C, device=device) * 0.1).to(DTYPE)

    # ---- CMix 权重 ----
    ffn_x_k = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    ffn_K = (torch.randn(C, FFN_DIM, device=device) * 0.3).to(DTYPE)
    ffn_V = (torch.randn(FFN_DIM, C, device=device) * 0.3).to(DTYPE)

    # ================================================================
    # 导出输入 & 状态
    # ================================================================
    save(OUT_DIR, 'x_in', x_in)
    save(OUT_DIR, 'tmix_x_prev.in', tmix_x_prev)
    save(OUT_DIR, 'cmix_x_prev.in', cmix_x_prev)
    save(OUT_DIR, 'v_first.in', v_first_in)
    save(OUT_DIR, 'att_state.in', att_state)

    # 别名: tmix/cmix 独立 TB 使用的文件名
    save(OUT_DIR, 'att.x_prev.in', tmix_x_prev)
    save(OUT_DIR, 'att.v_first.in', v_first_in)
    save(OUT_DIR, 'att.state.in', att_state)
    save(OUT_DIR, 'ffn.x_prev.in', cmix_x_prev)

    # ================================================================
    # 导出 LN1 权重
    # ================================================================
    save(OUT_DIR, 'ln1.weight', ln1_w)
    save(OUT_DIR, 'ln1.bias', ln1_b)

    # ================================================================
    # 导出 TMix 权重
    # ================================================================
    for name, val in [('att.x_r', x_r), ('att.x_w', x_w), ('att.x_k', x_k),
                       ('att.x_v', x_v), ('att.x_a', x_a), ('att.x_g', x_g)]:
        save(OUT_DIR, name, val)

    save(OUT_DIR, 'att.receptance.weight', R_)
    save(OUT_DIR, 'att.key.weight', K_)
    save(OUT_DIR, 'att.value.weight', V_)
    save(OUT_DIR, 'att.output.weight', O_)

    save(OUT_DIR, 'att.w0', w0)
    save(OUT_DIR, 'att.w1', w1)
    save(OUT_DIR, 'att.w2', w2)
    save(OUT_DIR, 'att.a0', a0)
    save(OUT_DIR, 'att.a1', a1)
    save(OUT_DIR, 'att.a2', a2)
    save(OUT_DIR, 'att.v0', v0)
    save(OUT_DIR, 'att.v1', v1)
    save(OUT_DIR, 'att.v2', v2)
    save(OUT_DIR, 'att.g1', g1)
    save(OUT_DIR, 'att.g2', g2)

    save(OUT_DIR, 'att.k_k', k_k)
    save(OUT_DIR, 'att.k_a', k_a)
    save(OUT_DIR, 'att.r_k', r_k)
    save(OUT_DIR, 'att.ln_x.weight', att_ln_w)
    save(OUT_DIR, 'att.ln_x.bias', att_ln_b)

    # ================================================================
    # 导出 LN2 权重
    # ================================================================
    save(OUT_DIR, 'ln2.weight', ln2_w)
    save(OUT_DIR, 'ln2.bias', ln2_b)

    # ================================================================
    # 导出 CMix 权重
    # ================================================================
    save(OUT_DIR, 'ffn.x_k', ffn_x_k)
    save(OUT_DIR, 'ffn.key.weight', ffn_K)
    save(OUT_DIR, 'ffn.value.weight', ffn_V)

    # ================================================================
    # Forward: LN1
    # ================================================================
    ln1_out = F.layer_norm(x_in.float(), (C,),
                           weight=ln1_w.float(), bias=ln1_b.float()).to(DTYPE)
    save(OUT_DIR, 'ln1.out', ln1_out)

    # ================================================================
    # Forward: TMix (复用 generate_tmix_mini 的核心函数)
    # ================================================================
    x_prev = [tmix_x_prev.clone(), cmix_x_prev.clone()]
    v_first = v_first_in.clone()

    tmix_out, v_first = RWKV_x070_TMix_one_export(
        OUT_DIR, layer_id=0,
        x=ln1_out, x_prev=x_prev, v_first=v_first, state=att_state,
        x_r=x_r, x_w=x_w, x_k=x_k, x_v=x_v, x_a=x_a, x_g=x_g,
        w0=w0, w1=w1, w2=w2,
        a0=a0, a1=a1, a2=a2,
        v0=v0, v1=v1, v2=v2,
        g1=g1, g2=g2,
        k_k=k_k, k_a=k_a, r_k=r_k,
        R_=R_, K_=K_, V_=V_, O_=O_,
        ln_w=att_ln_w, ln_b=att_ln_b
    )

    # ================================================================
    # Forward: Residual 1
    # ================================================================
    x_after_att = (x_in.float() + tmix_out.float()).to(DTYPE)
    save(OUT_DIR, 'x_after_att', x_after_att)

    # ================================================================
    # Forward: LN2
    # ================================================================
    ln2_out = F.layer_norm(x_after_att.float(), (C,),
                           weight=ln2_w.float(), bias=ln2_b.float()).to(DTYPE)
    save(OUT_DIR, 'ln2.out', ln2_out)

    # ================================================================
    # Forward: CMix (复用 generate_cmix_mini 的核心函数)
    # ================================================================
    cmix_out = RWKV_x070_CMix_one_export(OUT_DIR, ln2_out, x_prev, ffn_x_k, ffn_K, ffn_V)

    # ================================================================
    # Forward: Residual 2
    # ================================================================
    x_out = (x_after_att.float() + cmix_out.float()).to(DTYPE)
    save(OUT_DIR, 'x_out', x_out)

    # ================================================================
    # 导出状态输出
    # ================================================================
    save(OUT_DIR, 'tmix_x_prev.out', x_prev[0])
    save(OUT_DIR, 'cmix_x_prev.out', x_prev[1])
    save(OUT_DIR, 'v_first.out', v_first)
    save(OUT_DIR, 'att_state.out', att_state)

    # 别名: tmix/cmix 独立 TB 使用的文件名
    save(OUT_DIR, 'att.x_prev.out', x_prev[0])
    save(OUT_DIR, 'ffn.x_prev.out', x_prev[1])

    print("\nDone!")


if __name__ == "__main__":
    main()
