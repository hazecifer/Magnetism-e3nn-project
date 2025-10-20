from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .elements import map_species_to_Z
from .geometry import build_radius_graph_pbc
from .features import unit_vectors, spherical_harmonics_dirs, gaussian_rbf, random_so3, compute_pretrain_targets

class PretrainJSONDataset(Dataset):
    def __init__(self,
                 json_data: list[dict],
                 cutoff: float = 6.0,
                 lmax: int = 2,
                 rbf_dim: int = 64,
                 sigma: float = 2.0,
                 augment_so3: bool = True):
        self.structs = json_data
        self.cutoff = cutoff
        self.lmax = lmax
        self.rbf_dim = rbf_dim
        self.sigma = sigma
        self.augment = augment_so3

    def __len__(self):
        return len(self.structs)

    def __getitem__(self, idx: int):
        d = self.structs[idx]
        lattice = torch.tensor(d["lattice"], dtype=torch.get_default_dtype())  # 3,3 (rows)
        pos = torch.tensor(d["positions"], dtype=torch.get_default_dtype())    # N,3
        species = d["species"]
        Z = torch.tensor(map_species_to_Z(species), dtype=torch.long)

        # 全局随机 SO(3) 增强：同时旋转 lattice 与 positions（行向量右乘 R^T）
        if self.augment:
            R = random_so3().to(lattice.dtype)
            lattice = lattice @ R.t()
            pos = pos @ R.t()

        edge_index, edge_vec, edge_dist = build_radius_graph_pbc(lattice, pos, self.cutoff)
        # 可能存在孤立结构导致无边；做个兜底
        if edge_index.numel() == 0:
            # 构造空特征，伪标签也置零（不会用于训练，因为无边几乎不会发生在合理 cutoff 下）
            N = pos.shape[0]
            E = 0
            out = dict(
                lattice=lattice, pos=pos, Z=Z,
                edge_index=torch.zeros((2, E), dtype=torch.long),
                edge_vec=torch.empty((E, 3), dtype=pos.dtype),
                edge_dist=torch.empty((E,), dtype=pos.dtype),
                edge_rbf=torch.empty((E, self.rbf_dim), dtype=pos.dtype),
                edge_sh=torch.empty((E, (self.lmax+1)**2), dtype=pos.dtype),
                m_hat=torch.zeros((N, 3), dtype=pos.dtype),
                e1_axis=torch.zeros((N, 3), dtype=pos.dtype),
            )
            return out

        unit_dir = unit_vectors(edge_vec)
        edge_sh = spherical_harmonics_dirs(self.lmax, unit_dir)                # E, sum_{l<=lmax}(2l+1)
        edge_rbf = gaussian_rbf(edge_dist, self.cutoff, self.rbf_dim).to(pos.dtype)

        m_hat, e1_axis = compute_pretrain_targets(edge_index, edge_vec, edge_dist, pos.shape[0], self.sigma)

        return dict(
            lattice=lattice, pos=pos, Z=Z,
            edge_index=edge_index, edge_vec=edge_vec, edge_dist=edge_dist,
            edge_rbf=edge_rbf, edge_sh=edge_sh,
            m_hat=m_hat, e1_axis=e1_axis
        )

def collate_graphs(batch: list[dict]):
    # 将变长图打包成一个批次；节点偏移修正边索引；记录 batch 向量
    out = {}
    N_cum = 0
    node_offsets = []
    for b in batch:
        node_offsets.append(N_cum)
        N_cum += b["pos"].shape[0]

    # 节点级
    pos = torch.cat([b["pos"] for b in batch], dim=0)
    Z = torch.cat([b["Z"] for b in batch], dim=0)
    m_hat = torch.cat([b["m_hat"] for b in batch], dim=0)
    e1_axis = torch.cat([b["e1_axis"] for b in batch], dim=0)
    batch_idx = torch.cat([torch.full((b["pos"].shape[0],), i, dtype=torch.long) for i, b in enumerate(batch)])

    # 边级
    edge_indices = []
    edge_vec = []
    edge_dist = []
    edge_rbf = []
    edge_sh = []
    for off, b in zip(node_offsets, batch):
        ei = b["edge_index"]
        edge_indices.append(ei + off)
        edge_vec.append(b["edge_vec"])
        edge_dist.append(b["edge_dist"])
        edge_rbf.append(b["edge_rbf"])
        edge_sh.append(b["edge_sh"])
    if len(edge_indices) > 0 and edge_indices[0].numel() > 0:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_vec = torch.cat(edge_vec, dim=0)
        edge_dist = torch.cat(edge_dist, dim=0)
        edge_rbf = torch.cat(edge_rbf, dim=0)
        edge_sh = torch.cat(edge_sh, dim=0)
    else:
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_vec = torch.empty((0,3), dtype=pos.dtype)
        edge_dist = torch.empty((0,), dtype=pos.dtype)
        edge_rbf = torch.empty((0,batch[0]["edge_rbf"].shape[1]), dtype=pos.dtype)
        edge_sh = torch.empty((0,batch[0]["edge_sh"].shape[1]), dtype=pos.dtype)

    out.update(dict(
        pos=pos, Z=Z, batch=batch_idx,
        edge_index=edge_index, edge_vec=edge_vec, edge_dist=edge_dist,
        edge_rbf=edge_rbf, edge_sh=edge_sh,
        m_hat=m_hat, e1_axis=e1_axis,
        # 逐结构保留 lattice（可选，某些评估/导出用）
        lattices=[b["lattice"] for b in batch],
    ))
    return out

# ---------- 微调数据集：读取 spins，构图与增强一致 ----------
class FinetuneJSONDataset(Dataset):
    def __init__(self,
                 json_path: Path | str,
                 cutoff: float = 6.0,
                 lmax: int = 2,
                 rbf_dim: int = 64,
                 augment_so3: bool = True,
                 random_time_reversal: bool = True):
        with open(json_path, "r") as f:
            self.structs = json.load(f)
        self.cutoff = cutoff
        self.lmax = lmax
        self.rbf_dim = rbf_dim
        self.augment = augment_so3
        self.random_tr = random_time_reversal

    def __len__(self):
        return len(self.structs)

    def __getitem__(self, idx: int):
        d = self.structs[idx]
        lattice = torch.tensor(d["lattice"], dtype=torch.get_default_dtype())  # 3,3
        pos = torch.tensor(d["positions"], dtype=torch.get_default_dtype())    # N,3
        species = d["species"]
        Z = torch.tensor(map_species_to_Z(species), dtype=torch.long)
        spins = torch.tensor(d.get("spins", np.zeros((len(species), 3)).tolist()),
                             dtype=torch.get_default_dtype())  # N,3

        # 全局 SO(3) 旋转增强：一致旋转 lattice/pos/spins
        if self.augment:
            R = random_so3().to(lattice.dtype)
            lattice = lattice @ R.t()
            pos = pos @ R.t()
            spins = spins @ R.t()  # 轴矢在 SO(3) 下也按 R 作用

        # 随机时反翻转标签（损失用 |dot|，不会影响期望值）
        if self.random_tr and torch.rand(()) < 0.5:
            spins = -spins

        # 归一化与掩码（零向量不监督）
        spin_norm = torch.linalg.norm(spins, dim=-1, keepdim=True)
        mask = (spin_norm.squeeze(-1) > 1e-8)
        spins_dir = torch.where(spin_norm > 1e-8,
                                spins / spin_norm.clamp_min(1e-8),
                                torch.zeros_like(spins))

        edge_index, edge_vec, edge_dist = build_radius_graph_pbc(lattice, pos, self.cutoff)
        if edge_index.numel() == 0:
            E = 0
            return dict(
                lattice=lattice, pos=pos, Z=Z,
                edge_index=torch.zeros((2, E), dtype=torch.long),
                edge_vec=torch.empty((E, 3), dtype=pos.dtype),
                edge_dist=torch.empty((E,), dtype=pos.dtype),
                edge_rbf=torch.empty((E, self.rbf_dim), dtype=pos.dtype),
                edge_sh=torch.empty((E, (self.lmax+1)**2), dtype=pos.dtype),
                spins_dir=spins_dir, spin_mask=mask
            )

        unit_dir = unit_vectors(edge_vec)
        edge_sh = spherical_harmonics_dirs(self.lmax, unit_dir)
        edge_rbf = gaussian_rbf(edge_dist, self.cutoff, self.rbf_dim).to(pos.dtype)

        return dict(
            lattice=lattice, pos=pos, Z=Z,
            edge_index=edge_index, edge_vec=edge_vec, edge_dist=edge_dist,
            edge_rbf=edge_rbf, edge_sh=edge_sh,
            spins_dir=spins_dir, spin_mask=mask
        )

def collate_finetune(batch: list[dict]):
    out = {}
    N_cum = 0
    node_offsets = []
    for b in batch:
        node_offsets.append(N_cum)
        N_cum += b["pos"].shape[0]

    pos = torch.cat([b["pos"] for b in batch], dim=0)
    Z = torch.cat([b["Z"] for b in batch], dim=0)
    spins_dir = torch.cat([b["spins_dir"] for b in batch], dim=0)
    spin_mask = torch.cat([b["spin_mask"] for b in batch], dim=0)
    batch_idx = torch.cat([torch.full((b["pos"].shape[0],), i, dtype=torch.long) for i, b in enumerate(batch)])

    edge_indices, edge_vec, edge_dist, edge_rbf, edge_sh = [], [], [], [], []
    for off, b in zip(node_offsets, batch):
        ei = b["edge_index"]
        edge_indices.append(ei + off)
        edge_vec.append(b["edge_vec"])
        edge_dist.append(b["edge_dist"])
        edge_rbf.append(b["edge_rbf"])
        edge_sh.append(b["edge_sh"])
    if len(edge_indices) > 0 and edge_indices[0].numel() > 0:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_vec = torch.cat(edge_vec, dim=0)
        edge_dist = torch.cat(edge_dist, dim=0)
        edge_rbf = torch.cat(edge_rbf, dim=0)
        edge_sh = torch.cat(edge_sh, dim=0)
    else:
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_vec = torch.empty((0,3), dtype=pos.dtype)
        edge_dist = torch.empty((0,), dtype=pos.dtype)
        edge_rbf = torch.empty((0,batch[0]["edge_rbf"].shape[1]), dtype=pos.dtype)
        edge_sh = torch.empty((0,batch[0]["edge_sh"].shape[1]), dtype=pos.dtype)

    out.update(dict(
        pos=pos, Z=Z, batch=batch_idx,
        edge_index=edge_index, edge_vec=edge_vec, edge_dist=edge_dist,
        edge_rbf=edge_rbf, edge_sh=edge_sh,
        spins_dir=spins_dir, spin_mask=spin_mask,
        lattices=[b["lattice"] for b in batch],
    ))
    return out
