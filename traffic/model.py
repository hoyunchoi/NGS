from typing import Self, cast

import torch
from torch import nn

from NGS.model import MLP, GraphNetworkLayer, Model
from traffic.data import IN_WINDOW, OUT_WINDOW


class TrafficModel(Model):
    def __init__(
        self, num_nodes: int, emb_dim: int, depth: int, dropout: float
    ) -> None:
        """
        state emb: in_window * [speed, cos(TID), sin(TID), cos(DIW), sin(DIW)]
        dt emb : null (empty), not used
        node emb: 0, 1, ..., N
        edge emb: 1 - weight
        glob emb: last TID, DIW

        Note)
        - TID: time in day
        - DIW: day in week
        """
        super().__init__()

        # Embedding dimensions
        state_emb_dim = 5 * emb_dim
        node_emb_dim = 1 * emb_dim
        edge_emb_dim = 1 * emb_dim
        tid_emb_dim = 1 * emb_dim
        diw_emb_dim = 1 * emb_dim

        # Encoders
        self.state_encoder = MLP(5 * IN_WINDOW, state_emb_dim, state_emb_dim)
        self.dt_encoder = nn.Identity()
        self.node_encoder = nn.Embedding(num_nodes, node_emb_dim)
        self.edge_encoder = MLP(1, edge_emb_dim, edge_emb_dim)
        self.glob_encoder = nn.ModuleDict(
            {"tid": nn.Embedding(288, tid_emb_dim), "diw": nn.Embedding(7, diw_emb_dim)}
        )

        # Concatenate same type
        node_emb_dim += state_emb_dim
        glob_emb_dim = tid_emb_dim + diw_emb_dim

        # Graph Network Layers
        self.gn_layers = nn.ModuleList(
            GraphNetworkLayer(node_emb_dim, edge_emb_dim, glob_emb_dim, dropout=dropout)
            for _ in range(depth)
        )

        # Decoders
        self.decoder = nn.ModuleList(
            MLP(node_emb_dim, node_emb_dim, 1, last=True) for _ in range(OUT_WINDOW)
        )

    def cache(
        self: Self,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> None:
        """
        Args
        node_attr: [BN, 1], int64
        edge_attr: [BE, 1]
        glob_attr: [B, 2]
        """
        self._cached["node"] = self.node_encoder(
            node_attr.squeeze()
        )  # [BN, node_emb_dim]

        self._cached["edge"] = self.edge_encoder(edge_attr)  # [BE, edge_emb_dim]

        self.glob_encoder = cast(nn.ModuleDict, self.glob_encoder)
        self._cached["glob"] = torch.cat(  # [B, glob_emb_dim]
            [
                self.glob_encoder["tid"](glob_attr[:, 0]),  # [B, tid_emb_dim]
                self.glob_encoder["diw"](glob_attr[:, 1]),  # [B, diw_emb_dim]
            ],
            dim=-1,
        )

    def forward(
        self: Self,
        state: torch.Tensor,
        dt: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        ptr: torch.LongTensor,
    ) -> torch.Tensor:
        """
        WARNING:
        self.cache should be called before running self.forward
        self.empty_cache should be called after running self.forward

        Args
        state: [BN, 5 * W_in]
        dt: [B, 0]
        edge_index: [2, BE]
        batch: [BN, ]
        ptr: [B+1,]

        Return
        delta_state: [BN, W_out], prediction of delta state.
                     next_state = state + delta_state
        """
        state_emb = self.state_encoder(state)  # [BN, state_emb_dim]
        dt_emb = self.dt_encoder(dt)  # [B, 0]
        node_emb = self._cached["node"]  # [BN, node_emb_dim]
        edge_emb = self._cached["edge"]  # [BE, edge_emb_dim]
        glob_emb = self._cached["glob"]  # [B, glob_emb_dim]

        # Concatenate same type
        node_emb = torch.cat(  # [BN, state_emb_dim + node_emb_dim]
            [state_emb, node_emb], dim=1
        )

        # Graph Network layers
        for gn_layer in self.gn_layers:
            node_emb, edge_emb, glob_emb = gn_layer(
                node_emb, edge_emb, glob_emb, edge_index, batch
            )

        # Decoding: [BN, node_emb] -> [BN, W_out]
        last_speed = state[:, IN_WINDOW - 1].unsqueeze(-1)  # [BN, 1]
        self.decoder = cast(nn.ModuleList, self.decoder)
        return last_speed + torch.cat([decoder(node_emb) for decoder in self.decoder], dim=-1)
