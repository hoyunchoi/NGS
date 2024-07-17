from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch_geometric.utils import scatter


def count_trainable_param(model: nn.Module) -> int:
    """Return number of trainable parameters of model"""

    def count_single_parameter(param: torch.nn.parameter.Parameter) -> int:
        """Return number of trainable parameters of a single parameter"""
        if not param.requires_grad:
            return 0
        return 2 * param.numel() if param.is_complex() else param.numel()

    return sum(count_single_parameter(param) for param in model.parameters())


def prune_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    return state_dict


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, last: bool = False
    ) -> None:
        """
        Create 2-layer MLP module of shape in_dim -> hidden -> out_dim
        All activations are GELU

        depth: number of hidden layers + 2
        last: if True, do not use activation to the final output
        """
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        ]
        if not last:
            layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GraphNetworkLayer(nn.Module):
    def __init__(
        self,
        node_emb_dim: int,
        edge_emb_dim: int,
        glob_emb_dim: int,
    ) -> None:
        super().__init__()
        edge_hidden_dim = 2 * node_emb_dim + edge_emb_dim + glob_emb_dim
        self.per_edge_update = MLP(edge_hidden_dim, edge_hidden_dim, edge_emb_dim)

        node_hidden_dim = node_emb_dim + edge_emb_dim + glob_emb_dim
        self.per_node_update = MLP(node_hidden_dim, node_hidden_dim, node_emb_dim)

        glob_hidden_dim = node_emb_dim + edge_emb_dim + glob_emb_dim
        self.glob_update = MLP(glob_hidden_dim, glob_hidden_dim, glob_emb_dim)

    def forward(
        self: Self,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        node_attr: [BN, node_emb_dim]
        edge_attr: [BE, edge_emb_dim]
        glob_attr: [B, glob_emb_dim]
        edge_index: [2, BE]
        batch: [BN, ]
        ptr: [B+1,], pointer to batch index

        Returns
        node_attr: [BN, node_emb_dim]
        edge_attr: [BE, edge_emb_dim]
        """
        source, target = edge_index  # [BE, ], [BE, ]
        num_nodes = len(node_attr)

        # Edge update: [BE, 2 * node_emb_dim + edge_emb_dim + glob_emb_dim] -> [BE, edge_emb_dim]
        edge_emb = self.per_edge_update(
            torch.cat(
                [
                    node_attr[source],
                    node_attr[target],
                    edge_attr,
                    glob_attr[batch][source],
                ],
                dim=1,
            )
        )

        # Aggregation edge to node: [BE, edge_emb_dim] -> [BN, edge_emb_dim]
        edge2node = scatter(
            edge_emb, index=target, dim=0, dim_size=num_nodes, reduce="sum"
        )

        # Node update: [BN, node_emb_dim + edge_emb_dim + glob_emb_dim] -> [BN, node_emb_dim]
        node_emb = self.per_node_update(
            torch.cat([node_attr, edge2node, glob_attr[batch]], dim=1)
        )

        # Aggregation edge to glob, node to glob:
        # [BN, edge_emb_dim] -> [B, edge_emb_dim], [BN, node_emb_dim] -> [B, node_emb_dim]
        edge2glob = scatter(edge2node, index=batch, dim=0, reduce="min")
        node2glob = scatter(node_emb, index=batch, dim=0, reduce="min")

        # Glob update: [B, node_emb_dim + edge_emb_dim + glob_emb_dim] -> [B, glob_emb_dim]
        glob_emb = self.glob_update(torch.cat([node2glob, edge2glob, glob_attr], dim=1))

        return node_emb, edge_emb, glob_emb


class OneEncoder(nn.Module):
    """
    When given input is empty with shape [N, 0], return ones with shape [N, emb_dim]
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim

    @torch.no_grad()
    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return x.new_ones(size=(len(x), self.emb_dim))


class Model(nn.Module, ABC):
    state_encoder: MLP
    dt_encoder: MLP
    node_encoder: MLP | OneEncoder | nn.Identity
    edge_encoder: MLP | OneEncoder | nn.Identity
    glob_encoder: MLP | OneEncoder | nn.Identity
    gn_layers: nn.ModuleList
    decoder: MLP

    # Cached embeddings for node, edge, glob coefficients
    # Only used for rollout prediction
    _cached: dict[str, torch.Tensor] = {}

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    def cache(
        self: Self,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
    ) -> None:
        """
        Cache embeddings for node, edge, glob coefficients
        Used for rollout predictions, since coefficient is constant

        Args
        node_attr: [BN, node_emb_dim]
        edge_attr: [BE, edge_emb_dim]
        glob_attr: [B, glob_emb_dim]
        """
        self._cached["node"] = self.node_encoder(node_attr)
        self._cached["edge"] = self.edge_encoder(edge_attr)
        self._cached["glob"] = self.glob_encoder(glob_attr)

    def empty_cache(self: Self) -> None:
        self._cached = {}

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
        state: [BN, state_dim], state of each nodes
        dt: [B, 1], delta time step for the prediction of next node
        edge_index: [2, BE], which node is connected to which node
        batch: [BN, ], batch index for each node
        ptr: [B+1,], pointer to batch index

        Return
        delta_state: [BN, state_dim], prediction of delta state.
                     next_state = state + delta_state
        """
        # Get embeddings
        state_emb = self.state_encoder(state)  # [BN, state_emb_dim]
        dt_emb = self.dt_encoder(dt)  # [B, dt_emb_dim]
        node_emb = self._cached["node"]  # [BN, node_emb_dim]
        edge_emb = self._cached["edge"]  # [BE, edge_emb_dim]
        glob_emb = self._cached["glob"]  # [B, glob_emb_dim]

        # Concatenate same type
        node_emb = torch.cat(  # [BN, state_emb_dim + node_emb_dim]
            [state_emb, node_emb], dim=1
        )
        glob_emb = torch.cat(  # [B, dt_emb_dim + glob_emb_dim]
            [dt_emb, glob_emb], dim=1
        )

        # Graph Network layers
        for gn_layer in self.gn_layers:
            node_emb, edge_emb, glob_emb = gn_layer(
                node_emb, edge_emb, glob_emb, edge_index, batch
            )

        # Decoding: [BN, node_emb_dim + state_emb_dim] -> [BN, state_dim]
        return state + self.decoder(node_emb)
