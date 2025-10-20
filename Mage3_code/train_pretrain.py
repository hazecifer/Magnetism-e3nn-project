# config
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from e3nn import o3

from o3spin.elements import NUM_ELEMENTS
from o3spin.datasets import PretrainJSONDataset, collate_graphs
from o3spin.model import EquivariantBackbone
from o3spin.losses import directional_loss, axis_loss_abs, mean_angle_deg_abs
from o3spin.plotting import save_loss_curves

lmax = 2
rbf_dim = 64
cutoff = 6.0

batch_size = 25  # 该变量在预训练不使用，保留一致性

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_ckpt = "pretrained_backbone.pt"

JSON_PATH = Path("/Users/hazecifer/Documents/Codeproject/Myprojects/25-9-9 MgGNN/Testment/pretrain_toy.json")

# 读取原始结构列表
with open(JSON_PATH, "r") as f:
    raw_structs = json.load(f)
print(f"Loaded {len(raw_structs)} structures from {JSON_PATH}")

# ---------- 构建 DataLoader（预训练集） ----------
cutoff = 6.0
lmax = 2
rbf_dim = 64
sigma = 2.0
augment_so3 = True

pretrain_ds = PretrainJSONDataset(raw_structs, cutoff=cutoff, lmax=lmax, rbf_dim=rbf_dim, sigma=sigma, augment_so3=augment_so3)
pretrain_loader = DataLoader(pretrain_ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_graphs, drop_last=False)

# 取一个 batch 做 sanity check
batch = next(iter(pretrain_loader))
print("Batch nodes:", batch["pos"].shape, "edges:", batch["edge_index"].shape[1])
print("edge_rbf:", batch["edge_rbf"].shape, "edge_sh:", batch["edge_sh"].shape)
print("m_hat:", batch["m_hat"].shape, "e1_axis:", batch["e1_axis"].shape)

# ---------- 预训练：train/val 循环 ----------
def train_one_epoch_pretrain(model, loader, optimizer, alpha=0.3, device="cuda"):
    model.train()
    total_loss = 0.0
    total_ang = 0.0
    n_nodes = 0
    for batch in loader:
        Z = batch["Z"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_rbf = batch["edge_rbf"].to(device)
        edge_sh = batch["edge_sh"].to(device)
        m_hat = batch["m_hat"].to(device)
        e1_axis = batch["e1_axis"].to(device)

        optimizer.zero_grad(set_to_none=True)
        m_pred, _ = model(Z, edge_index, edge_sh, edge_rbf)  # 仅用 1o 头

        L_dir = directional_loss(m_pred, m_hat)
        L_axis = axis_loss_abs(m_pred, e1_axis)
        loss = L_dir + alpha * L_axis

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs_nodes = m_pred.shape[0]
        total_loss += loss.item() * bs_nodes
        total_ang += mean_angle_deg_abs(m_pred.detach(), m_hat.detach()) * bs_nodes
        n_nodes += bs_nodes

    return total_loss / n_nodes, total_ang / n_nodes

@torch.no_grad()
def validate_pretrain(model, loader, alpha=0.3, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_ang = 0.0
    n_nodes = 0
    for batch in loader:
        Z = batch["Z"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_rbf = batch["edge_rbf"].to(device)
        edge_sh = batch["edge_sh"].to(device)
        m_hat = batch["m_hat"].to(device)
        e1_axis = batch["e1_axis"].to(device)

        m_pred, _ = model(Z, edge_index, edge_sh, edge_rbf)

        L_dir = directional_loss(m_pred, m_hat)
        L_axis = axis_loss_abs(m_pred, e1_axis)
        loss = L_dir + alpha * L_axis

        bs_nodes = m_pred.shape[0]
        total_loss += loss.item() * bs_nodes
        total_ang += mean_angle_deg_abs(m_pred, m_hat) * bs_nodes
        n_nodes += bs_nodes

    return total_loss / n_nodes, total_ang / n_nodes

# ---------- 构建 train/val DataLoader ----------
# 使用已经创建的 pretrain_ds；这里做简单切分
perm = torch.randperm(len(pretrain_ds)).tolist()
split = int(0.8 * len(pretrain_ds))
train_idx, val_idx = perm[:split], perm[split:]
train_subset = torch.utils.data.Subset(pretrain_ds, train_idx)
val_subset = torch.utils.data.Subset(pretrain_ds, val_idx)

train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_graphs)
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_graphs)

# ---------- 实例化与跑若干 epoch 验证可用 ----------
model = EquivariantBackbone(
    num_elements=NUM_ELEMENTS,           # 或 max(ELEMENT_Z.values()) + 1
    hidden_scalar=64,
    hidden_vec_o=32,                     # 原来的 1o 通道数
    hidden_vec_e=32,                     # 新增的 1e 通道数（可与 1o 相同）
    hidden_tensor_e=16,                  # 2e 通道数
    layers=3,
    lmax=lmax,
    rbf_dim=rbf_dim,
    radial_mlp_dim=64,
    dropout_p=0.0
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=True)

epochs = 5  # 先小跑验证流程
alpha = 0.3


#储存训练历史
tr_hist, val_hist = [], []

for ep in range(1, epochs + 1):
    tr_loss, tr_ang = train_one_epoch_pretrain(model, train_loader, optimizer, alpha=alpha, device=device)
    val_loss, val_ang = validate_pretrain(model, val_loader, alpha=alpha, device=device)
    scheduler.step(val_loss)
    print(f"[Pretrain][{ep:02d}/{epochs}] train loss={tr_loss:.4f} ang(deg)={tr_ang:.2f} | val loss={val_loss:.4f} ang(deg)={val_ang:.2f}")

    # 记录损失
    tr_hist.append(tr_loss)
    val_hist.append(val_loss)

for ep in range(1, epochs + 1):
    tr_loss, tr_ang = train_one_epoch_pretrain(model, train_loader, optimizer, alpha=alpha, device=device)
    val_loss, val_ang = validate_pretrain(model, val_loader, alpha=alpha, device=device)
    scheduler.step(val_loss)
    print(f"[Pretrain][{ep:02d}/{epochs}] train loss={tr_loss:.4f} ang(deg)={tr_ang:.2f} | val loss={val_loss:.4f} ang(deg)={val_ang:.2f}")

# 保存预训练权重（骨干 + polar头权重）
pretrain_ckpt = "pretrained_backbone.pt"
torch.save(model.state_dict(), pretrain_ckpt)
print("Saved:", pretrain_ckpt)

save_loss_curves(tr_hist, val_hist, out_png="pretrain_loss.png", out_csv="pretrain_losses.csv", title="Pretrain Loss")