import itertools
from typing import Self, cast

import torch
from NGS.model import MLP, GraphNetworkLayer, Model, OneEncoder
from torch import nn


class KuramotoModel(Model):
    def __init__(self, emb_dim: int = 32, depth: int = 2) -> None:
        """
        state emb: cos(phase), sin(phase)
        dt emb: dt
        node emb: omega
        edge emb: null (empty). One for all edges
        glob emb: normalized coupling
        """
        super().__init__()

        # Embedding dimensions
        state_emb_dim = 2 * emb_dim
        dt_emb_dim = 1 * emb_dim
        node_emb_dim = 1 * emb_dim
        edge_emb_dim = 1 * emb_dim
        glob_emb_dim = 1 * emb_dim

        # Encoders
        self.state_encoder = MLP(2, state_emb_dim, state_emb_dim)
        self.dt_encoder = MLP(1, dt_emb_dim, dt_emb_dim)
        self.node_encoder = MLP(1, node_emb_dim, node_emb_dim)
        self.edge_encoder = OneEncoder(edge_emb_dim)
        self.glob_encoder = MLP(1, glob_emb_dim, glob_emb_dim)

        # Concatenate same type
        node_emb_dim += state_emb_dim
        glob_emb_dim += dt_emb_dim

        # Graph Network Layers
        self.gn_layers = nn.ModuleList(
            GraphNetworkLayer(node_emb_dim, edge_emb_dim, glob_emb_dim)
            for _ in range(depth)
        )

        # Decoders
        self.decoder = MLP(node_emb_dim, node_emb_dim, 1, last=True)

        # Threshold for computing interactions
        self.threshold = 0.0

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
        state: [BN, 1], state of each nodes
        dt: [B, 1], delta time step for the prediction of next node
        edge_index: [2, E], which node is connected to which node
        batch: [BN, ], batch index for each node
        ptr: [B+1,], pointer to batch index

        Return
        delta_state: [BN, state_dim], prediction of delta state.
                     next_state = state + delta_state
        """
        # Approximate graph
        edge_index = self.get_edge_index_batch(state % (2.0 * torch.pi), ptr)  # [2, BE]
        edge_attr = state.new_empty((edge_index.shape[1], 0))  # [BE, 0]

        # Overwrite edge embedding: just ones, without gradient
        self._cached["edge"] = self.edge_encoder(edge_attr)  # [BE, edge_emb_dim]

        # Change state to cos/sin representation: [BN, 2]
        state = torch.cat([state.cos(), state.sin()], dim=-1)

        return super().forward(state, dt, edge_index, batch, ptr)

    @torch.no_grad()
    def get_edge_index(self: Self, phase: torch.Tensor) -> torch.LongTensor:
        """
        Create adjacency matrix where sin(phase_i - phase_j) > sin(threshold)

        phase: [N, 1]

        Return
        edge_index: [2, E]
        """
        num_nodes = len(phase)
        phase_diff = (phase - phase.T).abs()  # [N, N], in the range of [0, 2pi]

        is_connected = torch.logical_or(
            torch.isclose(
                phase_diff,
                torch.full_like(phase_diff, 0.5 * torch.pi),
                rtol=0.0,
                atol=self.threshold,
            ),
            torch.isclose(
                phase_diff,
                torch.full_like(phase_diff, 1.5 * torch.pi),
                rtol=0.0,
                atol=self.threshold,
            ),
        )
        is_connected[torch.arange(num_nodes), torch.arange(num_nodes)] = False

        return cast(torch.LongTensor, is_connected.nonzero().T)

    @torch.no_grad()
    def get_edge_index_batch(
        self: Self, phase: torch.Tensor, ptr: torch.LongTensor
    ) -> torch.LongTensor:
        """
        phase: [BN, 1]
        ptr: [B+1, ]

        Returns
        edge_index: [2, BE]
        """
        # Iterate over graphs
        edge_indicies: list[torch.Tensor] = []
        for start, end in itertools.pairwise(ptr):
            edge_index = self.get_edge_index(phase[start:end])
            edge_indicies.append(edge_index + start)  # Increment count

        # Concatenate edges
        return cast(torch.LongTensor, torch.cat(edge_indicies, dim=-1))
