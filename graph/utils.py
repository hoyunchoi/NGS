from collections import Counter
from typing import cast, overload

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx.classes.reportviews import DegreeView


# ------------------ Networkx dependent ------------------
def filter_gcc(graph: nx.Graph) -> nx.Graph:
    gcc_nodes = sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    gcc = graph.subgraph(gcc_nodes)
    return nx.relabel.convert_node_labels_to_integers(gcc)


def get_mean_degree(graph: nx.Graph) -> float:
    return sum(d for _, d in cast(DegreeView, graph.degree)) / graph.number_of_nodes()


def get_degree_distribution(graph: nx.Graph) -> Counter[int]:
    degrees = [d for _, d in cast(DegreeView, graph.degree)]
    return Counter(degrees)


def get_edge_list(graph: nx.Graph) -> npt.NDArray[np.int64]:
    """
    Return edge list of shape (E, 2)\\
    If (0,1) is included in the edge list, (1,0) is not included
    """
    return np.array(graph.edges, dtype=np.int64)


@overload
def edge_list_2_adjacency_matrix(
    edge_list: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float32],
    num_nodes: int | None,
) -> npt.NDArray[np.float32]: ...


@overload
def edge_list_2_adjacency_matrix(
    edge_list: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float64],
    num_nodes: int | None,
) -> npt.NDArray[np.float64]: ...


def edge_list_2_adjacency_matrix(
    edge_list: npt.NDArray[np.int64],
    weights: np.ndarray,
    num_nodes: int | None = None,
) -> np.ndarray:
    """
    edge list: (E, 2). If (0,1) is included in the edge list, (1,0) is not included\\
    weights: (E, ) or (E, 1)
    """
    if num_nodes is None:
        num_nodes = int(edge_list.max()) + 1
    if weights.ndim == 2:
        assert weights.shape[1] == 1

    weighted_adj_matrix = np.zeros((num_nodes, num_nodes), dtype=weights.dtype)

    for (node1, node2), weight in zip(edge_list, weights.squeeze()):
        weighted_adj_matrix[node1, node2] = weight
        weighted_adj_matrix[node2, node1] = weight
    return weighted_adj_matrix


@overload
def get_weighted_adjacency_matrix(
    graph: nx.Graph, weights: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]: ...


@overload
def get_weighted_adjacency_matrix(
    graph: nx.Graph, weights: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]: ...


@overload
def get_weighted_adjacency_matrix(
    graph: nx.Graph, weights: npt.NDArray[np.float64] | float | None = None
) -> npt.NDArray[np.float64]: ...


def get_weighted_adjacency_matrix(
    graph: nx.Graph, weights: np.ndarray | float | None = None
) -> np.ndarray:
    """
    Return weighted adjacency matrix of shape [N, N].

    Args
    weights: (E, ) or (E, 1) if ndarray, assign for each edge
            float if constant over all edges
            None if 1 over all edges

    Return: weighted adjacency matrix
        If (i,j) is connected weight w, A[i,j] = A[j,i]=w
        If (i,j) is disconnected, A[i,j] = A[j,i] = 0
    Default dtype is float64
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    if weights is None:
        weights = np.ones(num_edges)
    elif isinstance(weights, float):
        weights = np.full(num_edges, weights)

    weights = cast(np.ndarray, weights)
    return edge_list_2_adjacency_matrix(get_edge_list(graph), weights, num_nodes)


@overload
def get_weighted_laplacian_matrix(
    graph: nx.Graph, weights: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]: ...


@overload
def get_weighted_laplacian_matrix(
    graph: nx.Graph, weights: npt.NDArray[np.float64] | float | None
) -> npt.NDArray[np.float64]: ...


def get_weighted_laplacian_matrix(
    graph: nx.Graph, weights: np.ndarray | float | None = None
) -> np.ndarray:
    """
    Return weighted laplacian matrix

    weights: (E, ) or (E, 1) if ndarray, assign for each edge\\
             float if constant over all edges\\
             None if 1 over all edges\\
    Default dtype is np.float64
    """
    weighted_adj_matrix = get_weighted_adjacency_matrix(graph, weights)
    dtype = weighted_adj_matrix.dtype

    degree_matrix = np.diag(np.sum(weighted_adj_matrix, axis=0)).view(dtype)
    return (degree_matrix - weighted_adj_matrix).view(dtype)


# --------------------- For pyg -------------------------
def directed2undirected(edge_list: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """
    Get directed edge list of shape (E, 2)\\
    Return undirected edge index of shape (2, 2E), for torch_geometric
    """
    return np.concatenate([edge_list, np.flip(edge_list, 1)]).T


@overload
def repeat_weight(weights: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...


@overload
def repeat_weight(weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


def repeat_weight(weights: np.ndarray) -> np.ndarray:
    """
    repeat the weight to change directed edge to undirected edge
    [E, ] -> [2E, ]
    [E, edge_dim] -> [2E, edge_dim]
    """
    return np.concatenate((weights, weights))
