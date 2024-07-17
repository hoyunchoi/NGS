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
from graph.utils import get_edge_list, get_weighted_adjacency_matrix
from path import DATA_DIR

arr32 = npt.NDArray[np.float32]
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

    # Graph args
    parser.add_argument("--num_nodes", type=int, nargs=2, default=[50, 100])
    parser.add_argument("--mean_degree", type=float, nargs=2, default=[2.0, 6.0])

    # Time args
    parser.add_argument("--max_time", type=float, nargs=2, default=(40.0, -1.0))
    parser.add_argument("--num_steps", type=int, nargs=2, default=(40, -1))
    parser.add_argument("--dt_delta", type=float, default=0.0)

    # Ensemble args
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    # Rossler args
    parser.add_argument("--a", type=float, nargs=2, default=[0.1, 0.3])
    parser.add_argument("--b", type=float, nargs=2, default=[0.1, 0.3])
    parser.add_argument("--c", type=float, nargs=2, default=[5.0, 7.0])
    parser.add_argument("--coupling", type=float, nargs=2, default=[0.02, 0.04])

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


def sample_position(num_nodes: int, rng: np.random.Generator) -> arr64:
    """
    Randomly sample initial position of each nodes
    Return: [N, 3], where x,y in [-4, 4] and z in [0, 6]
    """

    x = rng.uniform(-4.0, 4.0, size=num_nodes)
    y = rng.uniform(-4.0, 4.0, size=num_nodes)
    z = rng.uniform(0.0, 6.0, size=num_nodes)
    return np.stack([x, y, z], axis=1)  # [N, 3]


def sample_coupling(
    args_coupling: list[float], num_edges: int, rng: np.random.Generator
) -> arr64:
    """
    Randomly sample coupling constants of each edges
    Return: [E, ], Coupling constants of each edges
    """
    return rng.uniform(min(args_coupling), max(args_coupling), size=num_edges)


def sample_params(
    args_a: list[float],
    args_b: list[float],
    args_c: list[float],
    rng: np.random.Generator,
) -> arr64:
    """
    Randomly sample three parameters a,b,c
    Return: [3, ]
    """
    a = rng.uniform(min(args_a), max(args_a))
    b = rng.uniform(min(args_b), max(args_b))
    c = rng.uniform(min(args_c), max(args_c))

    return np.array([a, b, c])


@njit(fastmath=True)
def get_velocity(
    time: arr64, position: arr64, adjacency: arr64, params: arr64
) -> arr64:
    """
    Get velocity of each nodes in the coupled rossler system

    dx_i/dt = -y_i-z_i
    dy_i/dt = x_i + ay_i + sum_j A_ij (y_j-y_i)
            = x_i + ay_i + [Ay]_i - deg(i)*y_i
    dz_i/dt = b + z_i(x_i-c)

    Args
    time: [1, ], only for compatibility with scipy.integrate.solve_ivp
    position: [N, 3], position of each nodes
    adjacency: [N, N], Adjacency matrix of the graph, weighted by coupling constant
    params: [3, ], Parameters a, b, c

    Return
    velocity: [N, 3], velocity of each nodes
    """
    x, y, z = position.T
    y = np.ascontiguousarray(y)
    a, b, c = params

    velocity_x = -y - z
    velocity_y = x + a * y + adjacency @ y - adjacency.sum(axis=0) * y
    velocity_z = b + z * (x - c)

    return np.stack((velocity_x, velocity_y, velocity_z), axis=1)


def solve_rossler(
    adjacency: arr64, position: arr64, eval_time: arr64, params: arr64
) -> tuple[arr64, int, float]:
    """
    Solve coupled rossler equation

    Args
    adjacency: [N, N], Adjacency matrix of the graph, weighted by coupling constant
    position: [N, 3], Initial position of eahc node
    eval_time: [S+1, ], Evaluation time coordinates
    params: [3, ], Parameters a, b, c

    Returns
    trajectory: [S+1, N, 3], Position at each time step including initial condition
    num_func_evals: Number of function evaluations during solve_ivp
    runtime: Runtime
    """

    def step(time: arr64, state: arr64) -> arr64:
        """
        Single step dy/dt for solve_ivp

        Args
        time: [1, ], only for compatibility with scipy.integrate.solve_ivp
        state: [3N, ], concatenated position of each nodes
        """
        state = state.reshape(-1, 3)
        velocity = get_velocity(time, state, adjacency, params)
        return velocity.reshape(-1)

    start = time.perf_counter()
    result = solve_ivp(
        step,
        t_span=(eval_time[0], eval_time[-1]),
        y0=position.reshape(-1),
        method="DOP853",
        t_eval=eval_time,
        atol=1e-11,  # Around float64 precision
        rtol=1e-11,  # Around float64 precision
    )
    end = time.perf_counter()

    if result.status != 0:
        raise RuntimeError("Error in solve rossler equation")

    trajectory = result.y.T  # [S+1, 3N]
    trajectory = trajectory.reshape(len(trajectory), -1, 3)  # [S+1, N, 3]
    return trajectory, result.nfev, end - start


def run(args: argparse.Namespace, rng: np.random.Generator) -> SimulationData:
    # Sample variables for current simulation
    graph = sample_graph(args.num_nodes, args.mean_degree, rng)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    edge_list = get_edge_list(graph)  # [E, 2]

    eval_time = sample_eval_time(args.max_time, args.num_steps, args.dt_delta, rng)
    position = sample_position(num_nodes, rng)
    coupling = sample_coupling(args.coupling, num_edges, rng)
    params = sample_params(args.a, args.b, args.c, rng)

    # Simulate
    adjacency = get_weighted_adjacency_matrix(graph, coupling)
    trajectory, nfev, runtime = solve_rossler(adjacency, position, eval_time, params)

    # Store the result
    nx.set_node_attributes(  # [S+1, 3]
        graph, {node: trajectory[:, node] for node in range(num_nodes)}, "trajectory"
    )
    nx.set_node_attributes(  # [0, ]
        graph, {node: np.empty(0) for node in range(num_nodes)}, "node_attr"
    )
    nx.set_edge_attributes(
        graph,
        {tuple(edge): c.reshape(1) for edge, c in zip(edge_list, coupling)},
        "edge_attr",
    )

    return {
        "graph": graph,
        "eval_time": eval_time,  # [S+1, ]
        "glob_attr": params[None, :],  # [1, 3]
        "nfev": nfev,
        "runtime": runtime,
    }


def main() -> None:
    args = get_arguments()

    # Dummy run for compile
    _ = run(args, np.random.default_rng(args.seed))

    rng = np.random.default_rng(args.seed)
    simulation_data: list[SimulationData] = []
    for i in range(args.num_samples):
        print(i)
        data = run(args, rng)
        simulation_data.append(data)

    df = pd.DataFrame.from_records(simulation_data)
    df.to_pickle(DATA_DIR / f"{args.name}.pkl")


if __name__ == "__main__":
    main()
