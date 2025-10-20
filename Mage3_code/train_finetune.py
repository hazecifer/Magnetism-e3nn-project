# config
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from e3nn import o3

from o3spin.elements import NUM_ELEMENTS
from o3spin.datasets import FinetuneJSONDataset, collate_finetune
from o3spin.model import EquivariantBackbone
from o3spin.losses import spin_loss_abs, mean_angle_deg_abs_masked
from o3spin.utils import set_backbone_requires_grad, count_trainable_params, fmt
from o3spin.plotting import save_loss_curves  # 新增

lmax = 2
rbf_dim = 64
cutoff = 6.0

batch_size = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_ckpt = "pretrained_backbone.pt"

# ---------- 加载 finetune.json 并划分 ----------
FINETUNE_JSON_PATH = "/Users/hazecifer/Documents/Codeproject/Myprojects/25-9-9 MgGNN/Testment/train_toy.json"
print("Finetune json:", FINETUNE_JSON_PATH)

try:
    NUM_ELEMENTS
except NameError:
    NUM_ELEMENTS = NUM_ELEMENTS  # 已从包导入，保持原来结构的语义

# ---------- 构建数据加载器 ----------
finetune_ds = FinetuneJSONDataset(FINETUNE_JSON_PATH, cutoff=cutoff, lmax=lmax, rbf_dim=rbf_dim,
                                  augment_so3=True, random_time_reversal=True)
perm_ft = torch.randperm(len(finetune_ds)).tolist()
split_ft = int(0.8 * len(finetune_ds))
train_idx_ft, val_idx_ft = perm_ft[:split_ft], perm_ft[split_ft:]
train_ft = torch.utils.data.Subset(finetune_ds, train_idx_ft)
val_ft = torch.utils.data.Subset(finetune_ds, val_idx_ft)

train_loader_ft = DataLoader(train_ft, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_finetune)
val_loader_ft = DataLoader(val_ft, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_finetune)

# ---------- 构建模型并加载预训练骨干 ----------
model_ft = EquivariantBackbone(
    num_elements=NUM_ELEMENTS,
    hidden_scalar=64, hidden_vec_o=32, hidden_vec_e=32, hidden_tensor_e=16,
    layers=3, lmax=lmax, rbf_dim=rbf_dim, radial_mlp_dim=64, dropout_p=0.0
).to(device)

state = torch.load(pretrain_ckpt, map_location=device)
missing, unexpected = model_ft.load_state_dict(state, strict=False)
print("Loaded pretrain ckpt. missing:", missing, "unexpected:", unexpected)

# ---------- 微调训练循环 ----------
def train_one_epoch_finetune(model, loader, optimizer, device="cuda"):
    model.train()
    total_loss, total_ang, total_valid = 0.0, 0.0, 0
    skipped_no_supervision = 0
    skipped_no_grad = 0

    for batch in loader:
        Z = batch["Z"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_rbf = batch["edge_rbf"].to(device)
        edge_sh = batch["edge_sh"].to(device)
        spins_dir = batch["spins_dir"].to(device)
        spin_mask = batch["spin_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        _, s_pred = model(Z, edge_index, edge_sh, edge_rbf)  # 1e 头

        loss = spin_loss_abs(s_pred, spins_dir, spin_mask)
        if loss is None:
            skipped_no_supervision += 1
            continue
        if not loss.requires_grad:
            # 罕见：如果该 batch 的损失无 grad（比如全部上游冻结且路径被意外 detach），直接跳过，避免崩溃
            skipped_no_grad += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        valid_n = int(spin_mask.sum().item())
        total_loss += loss.item() * valid_n
        total_ang += mean_angle_deg_abs_masked(s_pred.detach(), spins_dir.detach(), spin_mask) * valid_n
        total_valid += valid_n

    if skipped_no_supervision > 0 or skipped_no_grad > 0:
        print(f"Note: skipped batches -> no_supervision={skipped_no_supervision}, no_grad={skipped_no_grad}")

    if total_valid == 0:
        return float("nan"), float("nan")
    return total_loss / total_valid, total_ang / total_valid

@torch.no_grad()
def validate_finetune(model, loader, device="cuda"):
    model.eval()
    total_loss, total_ang, total_valid = 0.0, 0.0, 0
    for batch in loader:
        Z = batch["Z"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_rbf = batch["edge_rbf"].to(device)
        edge_sh = batch["edge_sh"].to(device)
        spins_dir = batch["spins_dir"].to(device)
        spin_mask = batch["spin_mask"].to(device)

        _, s_pred = model(Z, edge_index, edge_sh, edge_rbf)
        loss = spin_loss_abs(s_pred, spins_dir, spin_mask)
        if loss is None:
            continue

        valid_n = int(spin_mask.sum().item())
        total_loss += loss.item() * valid_n
        total_ang += mean_angle_deg_abs_masked(s_pred, spins_dir, spin_mask) * valid_n
        total_valid += valid_n

    if total_valid == 0:
        return float("nan"), float("nan")
    return total_loss / total_valid, total_ang / total_valid

# 训练调度：前若干轮冻结骨干，再解冻继续训练
freeze_epochs = 5   # 如需先验证流程无误，可设为 0
epochs_total = 15   # 5 冻结 + 10 解冻示例

set_backbone_requires_grad(model_ft, False)
count_trainable_params(model_ft)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_ft.parameters()),
                              lr=3e-4, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=True)

# 新增：损失历史
tr_hist, val_hist = [], []

for ep in range(1, epochs_total + 1):
    if ep == freeze_epochs + 1:
        set_backbone_requires_grad(model_ft, True)
        count_trainable_params(model_ft)
        optimizer = torch.optim.AdamW(model_ft.parameters(), lr=2e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=True)

    tr_loss, tr_ang = train_one_epoch_finetune(model_ft, train_loader_ft, optimizer, device=device)
    val_loss, val_ang = validate_finetune(model_ft, val_loader_ft, device=device)
    if not math.isnan(val_loss):
        scheduler.step(val_loss)
    phase = "frozen" if ep <= freeze_epochs else "unfrozen"
    print(f"[Finetune:{phase}][{ep:02d}/{epochs_total}] train loss={fmt(tr_loss)} ang(deg)={fmt(tr_ang)} | "
          f"val loss={fmt(val_loss)} ang(deg)={fmt(val_ang)}")

    # 记录损失
    tr_hist.append(tr_loss)
    val_hist.append(val_loss)

# 保存微调权重
finetune_ckpt = "finetuned_spin.pt"
torch.save(model_ft.state_dict(), finetune_ckpt)
print("Saved:", finetune_ckpt)

# 新增：保存曲线
save_loss_curves(tr_hist, val_hist, out_png="finetune_loss.png", out_csv="finetune_losses.csv", title="Finetune Loss")
