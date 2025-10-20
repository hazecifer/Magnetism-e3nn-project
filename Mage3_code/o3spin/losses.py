import math
import torch

# ---------- 损失与评估 ----------

def directional_loss(u, v, eps=1e-8):
    # L_dir = 1 - u·v；u,v 已单位化
    dot = (u * v).sum(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return (1.0 - dot).mean()

def axis_loss_abs(u, v):
    # L_axis = 1 - |u·v|
    dot = (u * v).sum(-1).abs().clamp(0.0, 1.0)
    return (1.0 - dot).mean()

@torch.no_grad()
def mean_angle_deg_abs(u, v):
    # 角误差（度，|dot|）
    dot = (u * v).sum(-1).abs().clamp(0.0, 1.0)
    ang = torch.arccos(dot) * 180.0 / math.pi
    return ang.mean().item()

# ---------- 微调损失与指标（带掩码，空监督时返回 None） ----------
def spin_loss_abs(u_pred, v_true, mask):
    dot = (u_pred * v_true).sum(-1).abs().clamp(0.0, 1.0)  # (N,)
    loss = 1.0 - dot                                       # (N,)
    mask_f = mask.to(loss.dtype)                           # (N,)
    valid = int(mask_f.sum().item())
    if valid == 0:
        return None
    # 显式构造带 grad 的标量
    loss = (loss * mask_f).sum() / mask_f.sum()
    return loss

@torch.no_grad()
def mean_angle_deg_abs_masked(u_pred, v_true, mask):
    dot = (u_pred * v_true).sum(-1).abs().clamp(0.0, 1.0)
    ang = torch.arccos(dot) * 180.0 / math.pi
    return ang[mask].mean().item() if mask.any() else float("nan")
