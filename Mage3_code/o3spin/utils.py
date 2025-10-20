import math

def set_backbone_requires_grad(model, requires_grad: bool):
    for name, p in model.named_parameters():
        if name.startswith("polar_head") or name.startswith("spin_head"):
            continue
        p.requires_grad = requires_grad

def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")
    print("spin_head requires_grad:", [p.requires_grad for p in model.spin_head.parameters()])

def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) and not math.isnan(x) else "nan"
