import argparse
import sys
import time
from pathlib import Path
from typing import TypedDict

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit
from scipy.integrate import solve_ivp

sys.path.append(str(Path(__file__).parents[1]))
from graph.er import get_er
from graph.utils import get_edge_list, get_weighted_laplacian_matrix
from path import DATA_DIR

arr64 = npt.NDArray[np.float64]


class SimulationData(TypedDict):
    """
    N: number of nodes
    E: number of edges
    S: number of steps
    """

    graph: nx.Graph
    eval_time: npt.NDArray[np.float64]  # [S+1, ], including t=0
    glob_attr: npt.NDArray[np.float64]  # [glob_dim]
    nfev: int
    runtime: float


def get_arguments(options: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("name")

    # Graph config
    parser.add_argument("--num_nodes", type=int, nargs=2, default=[50, 100])
    parser.add_argument("--mean_degree", type=float, nargs=2, default=[2.0, 6.0])

    # Time config
    parser.add_argument("--max_time", type=float, nargs=2, default=(1.0, -1.0))
    parser.add_argument("--num_steps", type=int, nargs=2, default=(20, -1))
    parser.add_argument("--dt_delta", type=float, default=0.0)

    # Ensemble config
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    # Heat config
    parser.add_argument("--hot_ratio", type=float, nargs=2, default=[0.3, 0.7])
    parser.add_argument("--dissipation", type=float, nargs=2, default=[0.5, 2.0])

    return parser.parse_args(options)


def sample_graph(
    args_num_nodes: list[int],
    args_mean_degree: list[float],
    rng: np.random.Generator,
) -> nx.Graph:
    """
    Args
    num_nodes: number of nodes
    mean_degree: mean degree of the graph

    Return: graph with sampled number of nodes and mean degree.
    """
    num_nodes = rng.integers(min(args_num_nodes), max(args_num_nodes), endpoint=True)
    mean_degree = rng.uniform(min(args_mean_degree), max(args_mean_degree))

    graph = get_er(num_nodes, mean_degree, rng=rng)

    return graph


def divide_randomly(
    total: float, size: int, delta_percentile: float, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """
    Find array of floats with sum of total whose distribution is uniform

    Args
    total: sum(return) = total
    size: number of random numbers
    delta_percentile: return value lies between [total / size - delta, total / size + delta]
                      where delta = avg * delta_percentile

    Return
    floats: [size, ] where sum(floats) = total
    """
    assert 0.0 <= delta_percentile <= 1.0

    # Distribution setting
    avg = total / size
    delta = avg * delta_percentile
    low_range, high_range = avg - delta, avg + delta
    if total < size * low_range or total > size * high_range:
        raise ValueError("No sulution exists with given parameters.")

    # Find random floats with sum of total
    numbers = np.zeros(size)
    remaining = total
    for i in range(size):
        # Maximum/minimum range of current number
        high = min(high_range, remaining - (size - i - 1) * low_range)
        low = max(low_range, remaining - (size - i - 1) * high_range)

        # Randomly select number
        value = rng.uniform(min(low, high), max(low, high))
        numbers[i] = value
        remaining -= value

    rng.shuffle(numbers)
    return numbers


def sample_eval_time(
    args_max_time: tuple[float, float],
    args_num_steps: tuple[int, int],
    args_dt_delta: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    max_time1, max_time2 = args_max_time
    num_steps1, num_steps2 = args_num_steps

    # Evaluation time for the first interval
    dts = divide_randomly(max_time1, num_steps1, args_dt_delta, rng)
    eval_times1 = np.insert(np.cumsum(dts), 0, 0.0)
    eval_times1[-1] = max_time1  # Remove numerical error

    # If second interval is not given, return the first interval
    if max_time2 < 0:
        return eval_times1

    # Evaluation time for the second interval
    dts = divide_randomly(max_time2 - max_time1, num_steps2, args_dt_delta, rng)
    eval_times2 = np.cumsum(dts)
    eval_times2 += max_time1
    eval_times2[-1] = max_time2

    return np.concatenate([eval_times1, eval_times2])


def sample_temperature(
    args_hot_ratio: list[float], num_nodes: int, rng: np.random.Generator
) -> arr64:
    hot_ratio = rng.uniform(min(args_hot_ratio), max(args_hot_ratio))
    hot_spots = rng.choice(num_nodes, size=int(num_nodes * hot_ratio), replace=False)

    temperature = np.zeros(num_nodes)
    temperature[hot_spots] = 1.0
    return temperature


def sample_dissipation(
    args_dissipation: list[float], num_edges: int, rng: np.random.Generator
) -> arr64:
    return rng.uniform(min(args_dissipation), max(args_dissipation), size=num_edges)


def solve_heat(
    laplacian: arr64, temperature: arr64, eval_time: arr64
) -> tuple[arr64, int, float]:
    """
    Solve heat equation using Runge-Kutta method

    laplacian: [N, N], laplacian matrix of graph, weighted by dissipation rate
    temperature: [N, ], initial temperature
    eval_time: [S+1, ], Evaluation time coordinates

    Returns
    trajectory: [S+1, N, 1], temperature at each time step including initial condition
    num_func_evals: Number of function evaluations during solve_ivp
    runtime: Runtime
    """

    def step(time: arr64, state: arr64) -> arr64:
        """
        Single step dy/dt for solve_ivp

        Args
        time: [1, ], only for compatibility with scipy.integrate.solve_ivp
        state: [N, ], temperature of each node

        Return: [N, ], dissipation
        """
        return -np.dot(laplacian, state)

    start = time.perf_counter()
    result = solve_ivp(
        step,
        t_span=(eval_time[0], eval_time[-1]),
        y0=temperature,
        method="DOP853",
        t_eval=eval_time,
        atol=1e-11,  # Around float64 precision
        rtol=1e-11,  # Around float64 precision
    )
    end = time.perf_counter()

    if result.status != 0:
        raise RuntimeError("Error in solve heat equation")

    trajectory = result.y.T  # [S+1, N]
    return trajectory[..., None], result.nfev, end - start


@njit(fastmath=True)
def solve_heat_exact(
    laplacian: arr64, temperature: arr64, eval_time: arr64
) -> tuple[arr64, int, float]:
    """
    Solve heat equation exactly

    laplacian: [N, N], laplacian matrix of graph, weighted by dissipation rate
    temperature: [N, ], initial temperature of each node
    eval_time: [S+1, ], Evaluation time coordinates

    Return: [S+1, N, 1], temperature at each time step including initial condition
    """
    start = time.perf_counter()

    eig_val, eig_vec = np.linalg.eigh(laplacian)  # [N, ], [N, N]
    coeff = np.dot(temperature, eig_vec)  # [N, ]

    eig_val = eig_val.reshape(1, 1, -1)  # [1, 1, N]
    eig_vec = eig_vec.reshape(1, *eig_vec.shape)  # [1, N, N]
    coeff = coeff.reshape(1, 1, -1)  # [1, 1, N]
    eval_time = eval_time.reshape(-1, 1, 1)  # [S+1, 1, 1]

    trajectory = np.sum(  # [S+1, N]
        coeff * np.exp(-eig_val * eval_time) * eig_vec, axis=2
    )
    end = time.perf_counter()
    return trajectory[..., None], 0, end - start


def run(args: argparse.Namespace, rng: np.random.Generator) -> SimulationData:
    # Sample variables for current simulation
    graph = sample_graph(args.num_nodes, args.mean_degree, rng)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    edge_list = get_edge_list(graph)  # [E, 2]

    eval_time = sample_eval_time(args.max_time, args.num_steps, args.dt_delta, rng)
    temperature = sample_temperature(args.hot_ratio, num_nodes, rng)
    dissipation = sample_dissipation(args.dissipation, num_edges, rng)

    # Simulate
    laplacian = get_weighted_laplacian_matrix(graph, dissipation)
    trajectory, nfev, runtime = solve_heat(laplacian, temperature, eval_time)

    # Store the result
    nx.set_node_attributes(  # [S+1, 1]
        graph, {node: trajectory[:, node] for node in range(num_nodes)}, "trajectory"
    )
    nx.set_node_attributes(  # [0, ]
        graph, {node: np.empty(0) for node in range(num_nodes)}, "node_attr"
    )
    nx.set_edge_attributes(  # [1, ]
        graph,
        {tuple(edge): d.reshape(1) for edge, d in zip(edge_list, dissipation)},
        "edge_attr",
    )

    return {
        "graph": graph,
        "eval_time": eval_time.astype(np.float32),  # [S+1, ]
        "glob_attr": np.empty((1, 0)).astype(np.float32),  # [1, 0]
        "nfev": nfev,
        "runtime": runtime,
    }


def main() -> None:
    args = get_arguments()

    # Dummy run for compile
    _ = run(args, np.random.default_rng(args.seed))

    rng = np.random.default_rng(args.seed)
    simulation_data: list[SimulationData] = []
    for _ in range(args.num_samples):
        data = run(args, rng)
        simulation_data.append(data)

    df = pd.DataFrame.from_records(simulation_data)
    df.to_pickle(DATA_DIR / f"{args.name}.pkl")


if __name__ == "__main__":
    main()
