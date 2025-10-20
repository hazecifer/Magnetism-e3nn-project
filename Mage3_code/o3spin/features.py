import torch
from e3nn import o3

def gaussian_rbf(dist: torch.Tensor, cutoff: float, rbf_dim: int = 64, gamma: float | None = None) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, rbf_dim, dtype=dist.dtype, device=dist.device)
    if gamma is None and rbf_dim > 1:
        gamma = 1.0 / (centers[1] - centers[0]).pow(2)
    elif gamma is None:
        gamma = 10.0 / max(cutoff, 1e-6) ** 2
    d = dist.unsqueeze(-1) - centers  # E,D
    return torch.exp(-gamma * d.pow(2))

def unit_vectors(v: torch.Tensor, eps=1e-9) -> torch.Tensor:
    n = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return v / n

def spherical_harmonics_dirs(lmax: int, directions_E3: torch.Tensor) -> torch.Tensor:
    # Y_lm on unit sphere; returns shape (E, sum_{l<=lmax} (2l+1))
    return o3.spherical_harmonics(list(range(lmax+1)), directions_E3, normalize=True, normalization='component')

def random_so3() -> torch.Tensor:
    return o3.rand_matrix()  # 3x3

# 预训练自监督伪标签
def compute_pretrain_targets(edge_index: torch.Tensor,
                             edge_vec: torch.Tensor,
                             edge_dist: torch.Tensor,
                             num_nodes: int,
                             sigma: float):
    i, j = edge_index  # E
    w = torch.exp(-edge_dist / sigma)  # E

    # 主目标（极矢 1o）：m_i = normalize(sum_j w_ij r_ij)
    m = torch.zeros((num_nodes, 3), dtype=edge_vec.dtype, device=edge_vec.device)
    m = m.index_add(0, i, w.unsqueeze(1) * edge_vec)  # N,3
    m_norm = torch.linalg.norm(m, dim=1, keepdim=True).clamp_min(1e-8)
    m_hat = m / m_norm

    # 二阶矩 M_i = sum_j w_ij r r^T → 最大特征向量作为轴向 e1（1e）
    M = torch.zeros((num_nodes, 3, 3), dtype=edge_vec.dtype, device=edge_vec.device)
    outer = edge_vec.unsqueeze(-1) @ edge_vec.unsqueeze(-2)  # E,3,3
    M = M.index_add(0, i, w.view(-1, 1, 1) * outer)

    evals, evecs = torch.linalg.eigh(M)  # N,3,3 (升序)
    e1 = evecs[..., -1]  # N,3
    # 符号消歧：与 m_hat 保持朝向一致（m 很小时保持原样）
    dots = (e1 * m_hat).sum(-1, keepdim=True)
    sign = torch.where(m_norm > 1e-6, torch.sign(dots).where(dots.abs() > 1e-6, torch.ones_like(dots)), torch.ones_like(dots))
    e1 = e1 * sign
    e1 = unit_vectors(e1)

    return m_hat, e1  # m_hat: 1o 极矢方向；e1: 1e 轴向
