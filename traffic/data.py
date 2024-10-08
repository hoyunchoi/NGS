import itertools
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from NGS.data import NGSData, NGSDataset

arr32 = npt.NDArray[np.float32]

IN_WINDOW, OUT_WINDOW = 12, 12


class StandardScaler:
    """
    Standard scaler for given tensor of shape [..., C]
    The scaling is only done in the last dimension.

    Can adjust the standard deviation of transformed tensor by scale
    : N(0, scale^2)
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.avg = torch.tensor([])
        self.std = torch.tensor([])
        self.scale = scale

    def fit(self, data: npt.NDArray[np.float32] | torch.Tensor) -> None:
        data = torch.as_tensor(data).reshape(-1, data.shape[-1])
        self.std, self.avg = torch.std_mean(data, dim=0, unbiased=False)  # [C,], [C, ]

        # For channels that have all equal values, set std to 1.0 and avg to 0.0
        all_equal_mask = self.std == 0.0
        self.std[all_equal_mask] = 1.0
        self.avg[all_equal_mask] = 0.0

    def transform(self, data: npt.NDArray[np.float32] | torch.Tensor) -> torch.Tensor:
        shape = data.shape
        data = torch.as_tensor(data).reshape(-1, shape[-1])

        self.std, self.avg = self.std.to(data.device), self.avg.to(data.device)

        return self.scale * ((data - self.avg) / self.std).reshape(shape)

    def fit_transform(
        self, data: npt.NDArray[np.float32] | torch.Tensor
    ) -> torch.Tensor:
        """Do both fit and transform"""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(
        self, data: npt.NDArray[np.float32] | torch.Tensor
    ) -> torch.Tensor:
        shape = data.shape
        data = torch.as_tensor(data).reshape(-1, shape[-1])

        self.std, self.avg = self.std.to(data.device), self.avg.to(data.device)

        return (data / self.scale * self.std + self.avg).reshape(shape)

    def __str__(self) -> str:
        return f"Standard Scaler with mean: {self.avg}, std: {self.std}"

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"avg": self.avg, "std": self.std, "scale": torch.tensor(self.scale)}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.avg = state_dict["avg"]
        self.std = state_dict["std"]
        self.scale = state_dict["scale"].item()


def preprocess(
    df: pd.DataFrame,
    adj: arr32,
    in_window: int = IN_WINDOW,
    out_window: int = OUT_WINDOW,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> dict[str, Any]:
    """
    Preprocess the Traffic speeds
    N: number of nodes
    E: number of edges
    S: number of time steps

    Args:
    df: [S, N], dataframe containing speed of each node
    adj: [N, N], weighted adjacency matrix
    in_window: int, number of input time steps
    out_window: int, number of output time steps
    train_ratio: ratio of training speeds
    val_ratio: ratio of validation speeds

    Return:
    Dictionary containing the following keys
        - scaler: StandardScaler, scaler for normalization
        - trajectory: [S, N], normalized speed speeds
        - time_in_day: [S, ], time in day in a unit of 5 minuts
        - day_in_week: [S, ], day in week in a unit of day
        - indicies_train: [num_samples_train, 3], start, middle, end
        - indicies_val: [num_samples_val, 3], start, middle, end
        - indicies_test: [num_samples_test, 3], start, middle, end
        - edge_index: [2, E], edge index
        - edge_attr: [E, ], edge attribute
    """
    # Number of steps, samples
    trajectory = df.values.astype(np.float32)  # [num_steps, num_nodes]
    num_steps, num_nodes = trajectory.shape
    num_samples = num_steps - in_window - out_window + 1

    # Train, val, test split
    num_samples_train = round(num_samples * train_ratio)
    num_samples_val = round(num_samples * val_ratio)
    indicies = np.stack(  # start, middle, end indices
        [
            np.arange(num_samples),
            np.arange(num_samples) + in_window,
            np.arange(num_samples) + in_window + out_window,
        ],
        axis=-1,
    )
    indicies_train = indicies[:num_samples_train]
    indicies_val = indicies[num_samples_train : num_samples_train + num_samples_val]
    indicies_test = indicies[num_samples_train + num_samples_val :]

    # Normalize: fit on training speeds
    scaler = StandardScaler()
    scaler.fit(trajectory[: num_samples_train + in_window - 1].reshape(-1, 1))
    trajectory = (
        scaler.transform(trajectory.reshape(-1, 1)).reshape(trajectory.shape).numpy()
    )

    # Timestamps
    # Time in day in a unit of 5 minuts
    time_in_day = np.divide(
        df.index.values - df.index.values.astype("datetime64[D]"),
        np.timedelta64(5, "m"),
    ).astype(np.int64)

    # Day in week in a unit of day
    day_in_week = df.index.dayofweek.values.astype(np.int64)  # type: ignore

    # Graph
    edge_index, edge_attr = [], []
    adj -= np.eye(num_nodes, dtype=np.float32)  # remove self-loops
    for i, j in itertools.product(range(num_nodes), range(num_nodes)):
        if adj[i, j] != 0:
            edge_index.append([i, j])
            edge_attr.append(1.0 - adj[i, j])

    # Store
    return {
        "scaler": scaler,
        "trajectory": trajectory,
        "time_in_day": time_in_day,
        "day_in_week": day_in_week,
        "indicies_train": indicies_train,
        "indicies_val": indicies_val,
        "indicies_test": indicies_test,
        "edge_index": np.array(edge_index, dtype=np.int64).T,  # [2, E]
        "edge_attr": np.array(edge_attr, dtype=np.float32),  # [E, ]
    }


class TrafficDataset(NGSDataset):
    def __init__(
        self,
        trajectory: arr32,
        time_in_day: npt.NDArray[np.int64],
        day_in_week: npt.NDArray[np.int64],
        indicies: npt.NDArray[np.int64],
        edge_indices: npt.NDArray[np.int64],
        edge_attrs: arr32,
    ) -> None:
        """
        trajectory: [num_steps, num_nodes]
        time_in_day: [num_steps, ]
        day_in_week: [num_steps, ]
        indicies: [num_samples, 3], start, middle, end
        edge_index: [2, E]
        edge_attr: [E, ]
        """
        num_nodes = trajectory.shape[1]
        edge_index = cast(torch.LongTensor, torch.as_tensor(edge_indices))  # [2, E]
        edge_attr = torch.as_tensor(edge_attrs).unsqueeze(-1)  # [E, 1]
        node_attr = torch.arange(num_nodes).unsqueeze(-1)  # [N, 1]
        is_missing = cast(  # [N, ]
            torch.BoolTensor, torch.zeros(num_nodes, dtype=torch.bool)
        )
        dts = torch.empty(1, 1, 0)  # [1, 1, 0]

        self.data: list[NGSData] = []
        for idx in indicies:
            start, middle, end = idx

            # Treat nan_value speed as missing value. Replace with mean speed
            x = torch.tensor(trajectory[start:middle]).T  # [N, W_in]
            tid = torch.as_tensor(time_in_day[start:middle])  # [W_in], int64
            diw = torch.as_tensor(day_in_week[start:middle])  # [W_in], int64
            y = torch.tensor(trajectory[middle:end]).T

            tid_phase = (2 * np.pi * tid / 288).tile(num_nodes, 1)  # [N, W_in]
            diw_phase = (2 * np.pi * diw / 7).tile(num_nodes, 1)  # [N, W_in]
            x = torch.cat(  # [N, 5 * W_in]
                [x, tid_phase.cos(), tid_phase.sin(), diw_phase.cos(), diw_phase.sin()],
                dim=-1,
            )
            glob_attr = torch.stack([tid[-1], diw[-1]], dim=-1).unsqueeze(0)  # [1, 2]

            self.data.append(
                NGSData(
                    x=x,  # [N, 5 * W_in]
                    edge_index=edge_index,  # [2, E]
                    dts=dts,  # [1, 1, 0]
                    node_attr=node_attr,  # [N, 1]
                    edge_attr=edge_attr,  # [E, 1]
                    glob_attr=glob_attr,  # [1, 2]
                    is_missing=is_missing,  # [N, ]
                    y=y.unsqueeze(0),  # [1, N, W_out]
                )
            )
