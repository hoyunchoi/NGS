import networkx as nx
import numpy as np


def get_er(
    num_nodes: int,
    mean_degree: float,
    rng: np.random.Generator | int | None = None,
) -> nx.Graph:
    """Get giant component of ER random graph
    num_nodes: number of nodes. returning graph could be smaller
    mean_degree: mean degree of resulting graph
    """
    p = mean_degree / (num_nodes - 1)
    return nx.fast_gnp_random_graph(num_nodes, p, seed=rng)
