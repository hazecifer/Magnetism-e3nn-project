# 用一个样本做等变性测试：预测向量在全局旋转 R 下应满足 s' ≈ R·s, m' ≈ R·m
import torch
import math
from pathlib import Path

from o3spin.datasets import FinetuneJSONDataset
from o3spin.model import EquivariantBackbone
from o3spin.elements import NUM_ELEMENTS
from o3spin.geometry import build_radius_graph_pbc
from o3spin.features import unit_vectors, spherical_harmonics_dirs, gaussian_rbf, random_so3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 与训练一致的超参
lmax = 2
rbf_dim = 64
cutoff = 6.0

# 数据：取一个样本（避免增强与时反）
FINETUNE_JSON_PATH = "/Users/hazecifer/Documents/Codeproject/Myprojects/25-9-9 MgGNN/Testment/train_toy.json"
ds = FinetuneJSONDataset(FINETUNE_JSON_PATH, cutoff=cutoff, lmax=lmax, rbf_dim=rbf_dim,
                         augment_so3=False, random_time_reversal=False)
assert len(ds) > 0, "数据集为空"
sample = ds[0]

def forward_once(model, sample_dict):
    Z = sample_dict["Z"].to(device)
    edge_index = sample_dict["edge_index"].to(device)
    edge_rbf = sample_dict["edge_rbf"].to(device)
    edge_sh = sample_dict["edge_sh"].to(device)
    with torch.no_grad():
        m_pred, s_pred = model(Z, edge_index, edge_sh, edge_rbf)
    return m_pred, s_pred  # N,3

# 构建模型并加载已训练权重（优先用微调权重，其次预训练；若都无则用随机权重）
model = EquivariantBackbone(
    num_elements=NUM_ELEMENTS,
    hidden_scalar=64, hidden_vec_o=32, hidden_vec_e=32, hidden_tensor_e=16,
    layers=3, lmax=lmax, rbf_dim=rbf_dim, radial_mlp_dim=64, dropout_p=0.0
).to(device).eval()

ckpt = None
if Path("finetuned_spin.pt").exists():
    ckpt = "finetuned_spin.pt"
elif Path("pretrained_backbone.pt").exists():
    ckpt = "pretrained_backbone.pt"

if ckpt:
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded weights: {ckpt}")
else:
    print("No weights found, testing equivariance with randomly initialized model (仍应保持等变).")

# 原始预测
m0, s0 = forward_once(model, sample)

# 生成随机旋转，并对 lattice/pos 同步旋转后重建图与特征
R = random_so3().to(sample["pos"].dtype)  # 3x3
rot = sample.copy()
rot["lattice"] = sample["lattice"] @ R.t()
rot["pos"] = sample["pos"] @ R.t()
# 旋转不影响 Z
edge_index, edge_vec, edge_dist = build_radius_graph_pbc(rot["lattice"], rot["pos"], cutoff)
unit_dir = unit_vectors(edge_vec)
edge_sh = spherical_harmonics_dirs(lmax, unit_dir)
edge_rbf = gaussian_rbf(edge_dist, cutoff, rbf_dim).to(rot["pos"].dtype)

rot["edge_index"] = edge_index
rot["edge_vec"] = edge_vec
rot["edge_dist"] = edge_dist
rot["edge_sh"] = edge_sh
rot["edge_rbf"] = edge_rbf

# 旋转后的预测
m1, s1 = forward_once(model, rot)

# 将原始预测左乘 R，与旋转后预测比较
R_torch = R.to(device)
m0_R = (m0 @ R_torch.t())
s0_R = (s0 @ R_torch.t())

def angle_deg(u, v, eps=1e-8):
    # 返回每个节点的夹角（度）
    u = u / (u.norm(dim=-1, keepdim=True).clamp_min(eps))
    v = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))
    dot = (u * v).sum(-1).clamp(-1.0, 1.0)
    return (torch.arccos(dot) * 180.0 / math.pi)

ang_m = angle_deg(m1, m0_R)
ang_s = angle_deg(s1, s0_R)

print(f"Equiv test | polar(1o): mean={ang_m.mean().item():.4f} deg, max={ang_m.max().item():.4f} deg")
print(f"Equiv test | spin (1e): mean={ang_s.mean().item():.4f} deg, max={ang_s.max().item():.4f} deg")
print("N nodes:", m0.shape[0])
