from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch_geometric.data as gData
import torch_geometric.loader as gLoader
from graph.utils import directed2undirected, repeat_weight
from torch.utils.data import Dataset

arr32 = npt.NDArray[np.float32]
arr64 = npt.NDArray[np.float64]


def preprocess(
    df: pd.DataFrame, val_ratio: float = 1.0
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """
    Preprocess the dataframe to get the data for training and validation

    Ns: number of samples (graphs)
    N: number of nodes
    E: number of edges
    S: number of time steps

    Args
    df: dataframe of the data
    val_ratio: ratio of validation data

    Return
    Dictionary of train/validation data,
        edge_indices: Ns * [2, E]
        trajectories: Ns * [S+1, N, state_dim], including initial condition
        dts: Ns * [S, 1]
        node_attrs: Ns * [N, node_dim]
        edge_attrs: Ns * [E, edge_dim]
        glob_attrs: Ns * [1, glob_dim]
    """
    trajectories: list[arr32] = []  # [Ns, S+1, N, state_dim]
    node_attrs: list[arr32] = []  # [Ns, N, node_dim]
    edge_indices: list[npt.NDArray[np.int64]] = []  # [Ns, 2, E]
    edge_attrs: list[arr32] = []  # [Ns, E, edge_dim]

    for graph in df.graph:
        graph = cast(nx.Graph, graph)

        traj: list[arr64] = []  # N * [S+1, state_dim]
        na: list[arr64] = []  # N * [node_dim, ]
        for _, data in graph.nodes(data=True):
            traj.append(data["trajectory"])
            na.append(data["node_attr"])
        trajectory = np.stack(traj, axis=1, dtype=np.float32)  # [S+1, N, state_dim]
        node_attr = np.stack(na, dtype=np.float32)  # [N, node_dim]

        ei: list[npt.NDArray[np.int64]] = []  # E * [2, ]
        ea: list[arr64] = []  # E * [edge_dim, ]
        for node1, node2, data in graph.edges(data=True):
            ei.append(np.array([node1, node2], dtype=np.int64))
            ea.append(data["edge_attr"])
        if len(ei) == 0:
            # For Kuramoto model, there is no given edge
            # Create empty edge index and edge attribute
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, 0), dtype=np.float32)
        else:
            # Convert directed edge to undirected edge for pyg
            # Update edge attribute corresponding to the edge index
            edge_index = directed2undirected(np.array(ei, dtype=np.int64))
            edge_attr = repeat_weight(np.stack(ea, dtype=np.float32))

        trajectories.append(trajectory)  # Ns * [S+1, N, state_dim]
        node_attrs.append(node_attr)  # Ns * [N, node_dim]
        edge_indices.append(edge_index)  # Ns * [2, E]
        edge_attrs.append(edge_attr)  # Ns * [E, edge_dim]

    dts: list[arr32] = df.eval_time.map(  # Ns * [S, 1]
        lambda x: np.diff(x.astype(np.float32))[:, None]
    ).tolist()
    glob_attrs: list[arr32] = [  # Ns * [1, glob_dim]
        ga.astype(np.float32) for ga in df.glob_attr
    ]
    is_missings: list[npt.NDArray[np.bool_]] = [  # Ns * [N, ]
        np.zeros(trajectory.shape[1], dtype=np.bool_) for trajectory in trajectories
    ]

    # Train-val split
    num_vals = int(len(trajectories) * val_ratio)
    train_data = {
        "trajectories": trajectories[:-num_vals],
        "dts": dts[:-num_vals],
        "node_attrs": node_attrs[:-num_vals],
        "edge_attrs": edge_attrs[:-num_vals],
        "glob_attrs": glob_attrs[:-num_vals],
        "edge_indices": edge_indices[:-num_vals],
        "is_missings": is_missings[:-num_vals],
    }
    val_data = {
        "trajectories": trajectories[-num_vals:],
        "dts": dts[-num_vals:],
        "node_attrs": node_attrs[-num_vals:],
        "edge_attrs": edge_attrs[-num_vals:],
        "glob_attrs": glob_attrs[-num_vals:],
        "edge_indices": edge_indices[-num_vals:],
        "is_missings": is_missings[-num_vals:],
    }
    return train_data, val_data


def add_missing(
    train_data: dict[str, list[np.ndarray]],
    missing: float,
    seed: int,
    gn_depth: int = 2,
) -> None:
    """
    Choose missing values in the trajectory
    is_missings: [BN, ]. Missing nodes or nodes that are affected by missing nodes.
                         They will be masked out in loss calculation

    Args
    train_data: container of train data including edge_index
    missing: ratio of missing values
    seed: random seed
    gn_depth: depth of graph network layer in the NGS model
    """
    if missing == 0.0:
        # If there missing ratio is zero, do nothing
        return

    rng = np.random.default_rng(seed)
    for sample_idx in range(len(train_data["is_missings"])):
        is_missing = train_data["is_missings"][sample_idx]
        edge_index = train_data["edge_indices"][sample_idx]
        num_nodes = len(is_missing)

        # Number of missing nodes are too small
        missing_num = int(num_nodes * missing)
        if missing_num == 0:
            print("Missing ratio is too small. No missing data.")
            continue

        # Choose missing nodes
        missing_nodes = rng.choice(num_nodes, missing_num, replace=False)
        is_missing = np.zeros(num_nodes, dtype=np.bool_)  # [N, ]
        is_missing[missing_nodes] = True
        # train_data["trajectories"][sample_idx][:, is_missing, :] = np.nan

        # Trace nodes who are affected by missing nodes when using gn_depth
        # They will be masked out in loss calculation
        source, target = edge_index
        for _ in range(gn_depth):
            neighbors = target[np.isin(source, is_missing.nonzero()[0])]
            is_missing[neighbors] = True

        # Set missing nodes to nan
        train_data["is_missings"][sample_idx] = is_missing


def add_noise(train_data: dict[str, list[np.ndarray]], noise: float, seed: int) -> None:
    """
    Add noise to the trajectory in the train_data

    Args
    train_data: container of train data including edge_index
    noise: standard deviation of the Gaussian noise
    seed: random seed
    """
    if noise == 0.0:
        # If noise level is zero, do nothing
        return

    rng = np.random.default_rng(seed)

    for trajectory in train_data["trajectories"]:
        trajectory += rng.normal(0.0, noise, trajectory.shape).astype(trajectory.dtype)


class NGSData(gData.Data):
    x: torch.Tensor
    dts: torch.Tensor
    node_attr: torch.Tensor
    edge_attr: torch.Tensor
    glob_attr: torch.Tensor
    edge_index: torch.LongTensor
    is_missing: torch.BoolTensor
    y: torch.Tensor

    def __init__(
        self,
        x: torch.Tensor,
        dts: torch.Tensor,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        glob_attr: torch.Tensor,
        edge_index: torch.LongTensor,
        is_missing: torch.BoolTensor,
        y: torch.Tensor,
    ) -> None:
        """
        pyg Data instance for training of the NGS

        Args
        x: [N, state_dim], input state
        dts: [W, 1, 1], delta time steps
        node_attr: [N, node_dim], node coefficients
        edge_attr: [E, edge_dim], edge coefficients
        glob_attr: [1, glob_dim], global coefficients
        edge_index: [2, E], edge index
        is_missing: [N, ], missing nodes that should be masked out in loss
        y: [W, N, state_dim], next state(s)
        """
        super().__init__(
            x=x,
            edge_index=edge_index,
            dts=dts,
            node_attr=node_attr,
            edge_attr=edge_attr,
            glob_attr=glob_attr,
            y=y,
            is_missing=is_missing,
        )

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        """For member dts and y, concate"""
        if key in ["dts", "y"]:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class NGSDataset(Dataset):
    def __init__(
        self,
        trajectories: list[arr32],
        dts: list[arr32],
        node_attrs: list[arr32],
        edge_attrs: list[arr32],
        glob_attrs: list[arr32],
        edge_indices: list[npt.NDArray[np.int64]],
        is_missings: list[npt.NDArray[np.bool_]],
        window: int = 1,
    ) -> None:
        """
        Pytorch dataset containing NGSData instances

        Args
        trajectories: Ns * [S+1, N, state_dim]
        dts: Ns * [S, 1]
        node_attrs: Ns * [N, node_dim]
        edge_attrs: Ns * [E, edge_dim]
        glob_attrs: Ns * [1, glob_dim]
        edge_indices: Ns * [2, E]
        is_missings: Ns * [N, ]
        window: window size for single data
        """
        super().__init__()

        # Number of samples, time steps per sample, and window size
        num_samples, num_steps = len(trajectories), len(trajectories[0])
        window = window + num_steps if window < 0 else min(window, num_steps - 1)

        self.data: list[NGSData] = []
        for sample_idx in range(num_samples):
            trajectory = torch.as_tensor(  # [S+1, N, state_dim]
                trajectories[sample_idx]
            )
            dt = torch.as_tensor(dts[sample_idx])  # [S, 1]
            node_attr = torch.as_tensor(node_attrs[sample_idx])  # [N, node_dim]
            edge_attr = torch.as_tensor(edge_attrs[sample_idx])  # [E, edge_dim]
            glob_attr = torch.as_tensor(glob_attrs[sample_idx])  # [1, glob_dim]
            edge_index = cast(  # [2, E]
                torch.LongTensor, torch.as_tensor(edge_indices[sample_idx])
            )
            is_missing = cast(  # [N, ]
                torch.BoolTensor, torch.as_tensor(is_missings[sample_idx])
            )
            if is_missing.all():
                # All nodes are missing, do not use this sample
                continue

            for step in range(num_steps - window):
                ngs_data = NGSData(
                    x=trajectory[step],  # [N, state_dim]
                    dts=dt[step : step + window].unsqueeze(-1),  # [W, 1, 1]
                    node_attr=node_attr,  # [N, node_dim]
                    edge_attr=edge_attr,  # [E, edge_dim]
                    glob_attr=glob_attr,  # [1, glob_dim]
                    edge_index=edge_index,  # [2, E]
                    is_missing=is_missing,  # [N, ]
                    y=trajectory[step + 1 : step + window + 1],  # [W, N, state_dim]
                )
                self.data.append(ngs_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> NGSData:
        return self.data[idx]

    def to(self, device: torch.device) -> None:
        """Move All data instances to input device"""
        self.data = [data.to(str(device)) for data in self.data]


def get_loader(dataset: NGSDataset, **kwargs) -> gLoader.DataLoader:
    return gLoader.DataLoader(cast(gData.Dataset, dataset), **kwargs)
