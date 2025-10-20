import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3, nn as e3nn_nn

def sh_irreps(lmax: int) -> o3.Irreps:
    parts = [f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)]
    return o3.Irreps(" + ".join(parts))

class MessagePassingLayer(nn.Module):
    """
    等变消息传递层：TP(x_j, Y; w(edge_rbf)) → 聚合 → Gate
    隐藏表示包含 0e, 1o, 1e, 2e，Gate 同时门控 1o/1e/2e。
    """
    def __init__(self, irreps_hidden: o3.Irreps, edge_irreps: o3.Irreps,
                 rbf_dim: int,
                 hidden_scalar: int,
                 hidden_vec_o: int,
                 hidden_vec_e: int,
                 hidden_tensor_e: int,
                 radial_mlp_dim: int = 64,
                 use_batchnorm: bool = True,
                 dropout_p: float = 0.0):
        super().__init__()
        self.irreps_hidden = irreps_hidden
        self.edge_irreps = edge_irreps
        irreps_out = irreps_hidden

        # 动态权重 TP（每边权重）
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_hidden, edge_irreps, irreps_out, shared_weights=False
        )
        w_dim = self.tp.weight_numel

        self.radial = nn.Sequential(
            nn.Linear(rbf_dim, radial_mlp_dim),
            nn.SiLU(),
            nn.Linear(radial_mlp_dim, w_dim)
        )

        self.lin_res = o3.Linear(irreps_out, irreps_out)

        # Gate 配置
        self.irreps_scalars = o3.Irreps(f"{hidden_scalar}x0e")
        self.irreps_gated  = o3.Irreps(f"{hidden_vec_o}x1o + {hidden_vec_e}x1e + {hidden_tensor_e}x2e")
        self.irreps_gates  = o3.Irreps(f"{hidden_vec_o + hidden_vec_e + hidden_tensor_e}x0e")

        self.pre_gate = o3.Linear(
            irreps_out, self.irreps_scalars + self.irreps_gates + self.irreps_gated
        )
        # 这里用列表，长度等于各 irreps 的“项数”（len(Irreps)），不是通道总数
        self.gate = e3nn_nn.Gate(
            irreps_scalars=self.irreps_scalars,
            act_scalars=[nn.SiLU() for _ in range(len(self.irreps_scalars))],
            irreps_gates=self.irreps_gates,
            act_gates=[nn.Sigmoid() for _ in range(len(self.irreps_gates))],
            irreps_gated=self.irreps_gated
        )
        self.post_gate = o3.Linear(self.gate.irreps_out, irreps_out)

        # 若 e3nn_nn.BatchNorm 不可用，改为 o3.BatchNorm(irreps_out)
        self.bn = e3nn_nn.BatchNorm(irreps_out) if use_batchnorm else nn.Identity()
        self.dropout = nn.Identity() if dropout_p <= 0 else e3nn_nn.Dropout(irreps_out, p=dropout_p)

    def forward(self, x, edge_index, edge_sh, edge_rbf, num_nodes: int):
        i, j = edge_index
        w = self.radial(edge_rbf)         # (E, weight_numel)
        m_ij = self.tp(x[j], edge_sh, w)  # (E, hidden.dim)

        m_i = torch.zeros((num_nodes, m_ij.shape[-1]), dtype=m_ij.dtype, device=m_ij.device)
        m_i.index_add_(0, i, m_ij)

        h = x + self.lin_res(m_i)
        h = self.bn(h)
        h = self.pre_gate(h)
        h = self.gate(h)
        h = self.post_gate(h)
        h = self.dropout(h)
        return h

class EquivariantBackbone(nn.Module):
    """
    隐藏包含 0e + 1o + 1e + 2e；两个头：1o（预训练）与 1e（微调）
    """
    def __init__(self, num_elements: int,
                 hidden_scalar: int = 64,
                 hidden_vec_o: int = 32,
                 hidden_vec_e: int = 32,
                 hidden_tensor_e: int = 16,
                 layers: int = 3,
                 lmax: int = 2,
                 rbf_dim: int = 64,
                 radial_mlp_dim: int = 64,
                 dropout_p: float = 0.0):
        super().__init__()
        self.irreps_hidden = o3.Irreps(
            f"{hidden_scalar}x0e + {hidden_vec_o}x1o + {hidden_vec_e}x1e + {hidden_tensor_e}x2e"
        )
        self.edge_irreps = sh_irreps(lmax)  # 0e + 1o + 2e

        self.emb = nn.Embedding(num_elements, hidden_scalar)  # Z -> 0e
        self.in_lin = o3.Linear(o3.Irreps(f"{hidden_scalar}x0e"), self.irreps_hidden)

        self.layers = nn.ModuleList([
            MessagePassingLayer(self.irreps_hidden, self.edge_irreps, rbf_dim,
                                hidden_scalar, hidden_vec_o, hidden_vec_e, hidden_tensor_e,
                                radial_mlp_dim=radial_mlp_dim, use_batchnorm=True, dropout_p=dropout_p)
            for _ in range(layers)
        ])

        self.polar_head = o3.Linear(self.irreps_hidden, o3.Irreps("1o"))
        self.spin_head  = o3.Linear(self.irreps_hidden, o3.Irreps("1e"))

    def forward(self, Z, edge_index, edge_sh, edge_rbf):
        N = Z.shape[0]
        x0 = self.emb(Z)
        x = self.in_lin(x0)
        for layer in self.layers:
            x = layer(x, edge_index=edge_index, edge_sh=edge_sh, edge_rbf=edge_rbf, num_nodes=N)

        polar = self.polar_head(x).view(N, 3)
        spin  = self.spin_head(x).view(N, 3)

        def to_unit(v, eps=1e-9):
            n = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
            return v / n

        return to_unit(polar), to_unit(spin)
