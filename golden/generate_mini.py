"""
CMix mini golden data generator — 基于 RWKV_x070_CMix_one 改写，加入中间结果导出。

生成缩小维度 (C=8, FFN_DIM=32) 的 CMix 测试数据，
变量名和文件名与 export_activations.py 保持一致。

输出: golden/mini/cmix/ 下的 hex 文件
用法: cd C:/Projects/Albatross && uv run -m golden.generate_mini
"""

import os
import glob
import torch
from golden.utils import GOLDEN_ROOT, save
from reference.rwkv7 import DTYPE

C = 8
FFN_DIM = 32

OUT_DIR = os.path.join(GOLDEN_ROOT, "mini", "cmix")


# 从 RWKV_x070_CMix_one 复制，加入中间结果导出
def RWKV_x070_CMix_one_export(out_dir, x, x_prev, x_k, K_, V_):
    xx = x_prev[1] - x
    x_prev[1] = x

    k = x + xx * x_k
    save(out_dir, 'ffn.x_k.out', k)

    k_expanded = k @ K_
    save(out_dir, 'ffn.key.weight.in', k)
    save(out_dir, 'ffn.key.weight.out', k_expanded)

    k = torch.relu(k_expanded) ** 2
    save(out_dir, 'ffn.key.weight.act', k)

    save(out_dir, 'ffn.value.weight.in', k)
    out = k @ V_
    save(out_dir, 'ffn.value.weight.out', out)

    return out


def main():
    # 清空旧数据，避免残留
    if os.path.exists(OUT_DIR):
        for f in glob.glob(os.path.join(OUT_DIR, '*.hex')):
            os.remove(f)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(42)
    device = "cuda"

    print(f"Generating cmix mini golden data (C={C}, FFN_DIM={FFN_DIM})")
    print(f"Output: {OUT_DIR}/\n")

    # 生成随机输入 (小范围，避免 FP16 溢出)
    x = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    x_prev_cmix = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    x_k = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)  # [0.1, 0.9]

    # 权重矩阵 — PyTorch 布局 (模型加载时已转置)
    key_weight = (torch.randn(C, FFN_DIM, device=device) * 0.3).to(DTYPE)
    value_weight = (torch.randn(FFN_DIM, C, device=device) * 0.3).to(DTYPE)

    # 导出输入 & 状态 & 权重
    save(OUT_DIR, 'ln2.out', x)
    save(OUT_DIR, 'ffn.x_prev.in', x_prev_cmix)
    save(OUT_DIR, 'ffn.x_k', x_k)
    save(OUT_DIR, 'ffn.key.weight', key_weight)      # [C, FFN_DIM]
    save(OUT_DIR, 'ffn.value.weight', value_weight)   # [FFN_DIM, C]

    # 调用带导出的 CMix (中间结果在函数内部导出)
    x_prev = [None, x_prev_cmix.clone()]
    RWKV_x070_CMix_one_export(OUT_DIR, x, x_prev, x_k, key_weight, value_weight)

    # 状态输出 (x_prev[1] 已被函数更新为 x)
    save(OUT_DIR, 'ffn.x_prev.out', x_prev[1])

    print("\nDone!")


if __name__ == "__main__":
    main()
