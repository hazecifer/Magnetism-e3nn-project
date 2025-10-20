import torch

def cartesian_to_fractional(lattice_rows_3x3: torch.Tensor, pos_N3: torch.Tensor) -> torch.Tensor:
    # lattice 是行向量；r_cart = A^T f，所以 f = (A^T)^{-1} r
    inv_AT = torch.inverse(lattice_rows_3x3.t())
    return (inv_AT @ pos_N3.t()).t()

def fractional_to_cartesian(lattice_rows_3x3: torch.Tensor, frac_N3: torch.Tensor) -> torch.Tensor:
    # r_cart = A^T f
    return (lattice_rows_3x3.t() @ frac_N3.t()).t()

def build_radius_graph_pbc(lattice, pos, cutoff: float):
    # 最小镜像准则在分数坐标系中完成，再映回笛卡尔
    pos = pos.to(torch.get_default_dtype())
    lattice = lattice.to(torch.get_default_dtype())
    N = pos.shape[0]
    frac = cartesian_to_fractional(lattice, pos)  # N,3

    rows_i, rows_j, edge_vecs, edge_dists = [], [], [], []
    for i in range(N):
        df = frac - frac[i]                # N,3
        df = df - torch.round(df)          # 最小镜像 [-0.5,0.5]
        dr = fractional_to_cartesian(lattice, df)  # N,3
        dist = torch.linalg.norm(dr, dim=1)
        mask = (dist > 1e-8) & (dist <= cutoff)
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            rows_i.append(torch.full((idx.numel(),), i, dtype=torch.long))
            rows_j.append(idx.to(torch.long))
            edge_vecs.append(dr[idx])
            edge_dists.append(dist[idx])

    if len(rows_i) == 0:
        # 无边时返回空张量
        E0 = torch.zeros((2,0), dtype=torch.long)
        return E0, torch.empty((0,3), dtype=pos.dtype), torch.empty((0,), dtype=pos.dtype)

    edge_i = torch.cat(rows_i); edge_j = torch.cat(rows_j)
    edge_index = torch.stack([edge_i, edge_j], dim=0)  # 2,E
    edge_vec = torch.cat(edge_vecs, dim=0)             # E,3
    edge_dist = torch.cat(edge_dists, dim=0)           # E,
    return edge_index, edge_vec, edge_dist
