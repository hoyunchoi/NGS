from NGS.model import MLP, GraphNetworkLayer, Model
from torch import nn


class RosslerModel(Model):
    def __init__(self, emb_dim: int = 32, depth: int = 2) -> None:
        """
        state emb: x, y, z
        dt emb: dt
        node emb: null (empty tensor)
        edge emb: dissipation rate
        glob emb: a, b, c
        """
        super().__init__()

        # Embedding dimensions
        state_emb_dim = 3 * emb_dim
        dt_emb_dim = 1 * emb_dim
        node_emb_dim = 0 * emb_dim
        edge_emb_dim = 1 * emb_dim
        glob_emb_dim = 3 * emb_dim

        # Encoders
        self.state_encoder = MLP(3, state_emb_dim, state_emb_dim)
        self.dt_encoder = MLP(1, dt_emb_dim, dt_emb_dim)
        self.node_encoder = nn.Identity()
        self.edge_encoder = MLP(1, edge_emb_dim, edge_emb_dim)
        self.glob_encoder = MLP(3, glob_emb_dim, glob_emb_dim)

        # Concatenate same type
        node_emb_dim += state_emb_dim
        glob_emb_dim += dt_emb_dim

        # Graph Network Layers
        self.gn_layers = nn.ModuleList(
            GraphNetworkLayer(node_emb_dim, edge_emb_dim, glob_emb_dim)
            for _ in range(depth)
        )

        # Decoders
        self.decoder = MLP(node_emb_dim, node_emb_dim, 3, last=True)
