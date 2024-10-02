from typing import Self
import torch
from torch import nn

from NGS.model import MLP, GraphNetworkLayer, Model


class HeatModel(Model):
    def __init__(self, emb_dim: int, depth: int, dropout: float) -> None:
        """
        state emb: temperature
        dt emb: dt
        node emb: null (empty tensor)
        edge emb: dissipation rate
        glob emb: null (empty tensor)
        """
        super().__init__()

        # Embedding dimensions
        state_emb_dim = 1 * emb_dim
        dt_emb_dim = 1 * emb_dim
        node_emb_dim = 0 * emb_dim
        edge_emb_dim = 1 * emb_dim
        glob_emb_dim = 0 * emb_dim

        # Encoders
        self.state_encoder = MLP(1, state_emb_dim, state_emb_dim)
        self.dt_encoder = MLP(1, dt_emb_dim, dt_emb_dim)
        self.node_encoder = nn.Identity()
        self.edge_encoder = MLP(1, edge_emb_dim, edge_emb_dim)
        self.glob_encoder = nn.Identity()

        # Concatenate same type
        node_emb_dim += state_emb_dim
        glob_emb_dim += dt_emb_dim

        # Graph Network Layers
        self.gn_layers = nn.ModuleList(
            GraphNetworkLayer(node_emb_dim, edge_emb_dim, glob_emb_dim, dropout)
            for _ in range(depth)
        )

        # Decoders
        self.decoder = MLP(node_emb_dim, node_emb_dim, 1, last=True)

    def forward(
        self: Self,
        state: torch.Tensor,
        dt: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        ptr: torch.LongTensor,
    ) -> torch.Tensor:
        return state + super().forward(state, dt, edge_index, batch, ptr)
