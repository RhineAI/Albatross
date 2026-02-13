import torch
from torch.nn import functional as F

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # 基础 tensor 运算
    a = torch.randn(3, 3, device="cuda", dtype=torch.float16)
    b = torch.randn(3, 3, device="cuda", dtype=torch.float16)
    c = a @ b
    print(f"\nMatmul on GPU: {c.shape} {c.dtype} -> OK")

    # layer_norm
    x = torch.randn(1, 64, device="cuda", dtype=torch.float16)
    w = torch.ones(64, device="cuda", dtype=torch.float16)
    b = torch.zeros(64, device="cuda", dtype=torch.float16)
    y = F.layer_norm(x, (64,), weight=w, bias=b)
    print(f"LayerNorm: {y.shape} -> OK")

    # softmax
    logits = torch.randn(1, 65536, device="cuda", dtype=torch.float16)
    probs = F.softmax(logits.float(), dim=-1)
    print(f"Softmax: {probs.shape}, sum={probs.sum().item():.4f} -> OK")

    # group_norm
    x = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    y = F.group_norm(x.view(1, 128), num_groups=2)
    print(f"GroupNorm: {y.shape} -> OK")

    # relu squared
    x = torch.randn(64, device="cuda", dtype=torch.float16)
    y = torch.relu(x) ** 2
    print(f"ReLU^2: {y.shape} -> OK")

    # sigmoid
    x = torch.randn(64, device="cuda", dtype=torch.float16)
    y = torch.sigmoid(x)
    print(f"Sigmoid: {y.shape} -> OK")

    # topk
    logits = torch.randn(65536, device="cuda")
    _, indices = torch.topk(logits, 5)
    print(f"TopK: {indices.tolist()} -> OK")

    print("\nAll tests passed!")
else:
    print("No CUDA GPU available.")
