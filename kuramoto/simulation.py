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

    # Graph args
    parser.add_argument("--num_nodes", type=int, nargs=2, default=[50, 100])
    parser.add_argument("--threshold", type=float, default=2.0)

    # Time args
    parser.add_argument("--max_time", type=float, default=10.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--dt_delta", type=float, default=0.0)

    # Ensemble args
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    # Kuramoto args
    parser.add_argument("--coupling", type=float, nargs=2, default=[0.3, 0.7])

    return parser.parse_args(options)


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


def sample_phase(num_nodes: int, rng: np.random.Generator) -> arr64:
    """
    Randomly sample intial phase of each node
    Return: [N, ] in the range of [-pi, pi]
    """
    return rng.uniform(-np.pi, np.pi, size=num_nodes)


def sample_omega(num_nodes: int, rng: np.random.Generator) -> arr64:
    """
    Randomly sample omega (natural angular velocity) of each nodes
    Return: [N, ], Omega of each node, clipped to [-3, 3]
    """
    return np.clip(rng.normal(size=num_nodes), -3.0, 3.0)


def sample_coupling(args_coupling: list[float], rng: np.random.Generator) -> float:
    """Randomly sample global coupling constant"""
    return rng.uniform(min(args_coupling), max(args_coupling))


@njit(fastmath=True)
def get_adjacency(phase: arr64, threshold: float) -> arr64:
    """
    phase: [N, ]
    """
    phase = phase[:, None] % (2.0 * np.pi)  # [N, 1] in the range of [0, 2pi]
    phase_diff = np.abs(phase - phase.T)  # [N, N] in the range of [0, 2pi]

    adjacency = (np.abs(phase_diff - 0.5 * np.pi) < threshold) + (
        np.abs(phase_diff - 1.5 * np.pi) < threshold
    )
    adjacency = adjacency & ~np.eye(len(adjacency), dtype=np.bool_)
    return adjacency.astype(np.float64)


@njit(fastmath=True)
def get_velocity(
    time: arr64,
    phase: arr64,
    normalized_coupling: float,
    omega: arr64,
    threshold: float = 2.0,
) -> arr64:
    """
    Get angular velocity of each node.
    velocity_i = omega_i + sum_j K_ij sin(phase_j - phase_i)
               = omega_i + cos(phase_i) sum_j K_ij sin(phase_j) - sin(phase_i) sum_j K_ij cos(phase_j)
    Args
    time: [1, ], only for compatibility with scipy.integrate.solve_ivp
    phase: [N, ], phase of each node
    adjacency: [N, N], adjacency matrix of graph, weighted by coupling constant K
    omega: [N, ], omega of each node

    Return: [N, ], delta phase
    """
    num_nodes = len(phase)

    # Compute adjacency matrix according to the threshold
    adjacency = (
        get_adjacency(phase, threshold)
        if threshold < 0.5 * np.pi
        else np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    )
    weighted_adjacency = normalized_coupling * adjacency

    sin_phase, cos_phase = np.sin(phase), np.cos(phase)

    return (
        omega
        + cos_phase * np.dot(weighted_adjacency, sin_phase)
        - sin_phase * np.dot(weighted_adjacency, cos_phase)
    )


@njit(fastmath=True)
def get_jacobian(
    time: arr64,
    phase: arr64,
    normalized_coupling: float,
    omega: arr64,
    threshold: float = 2.0,
) -> arr64:
    num_nodes = len(phase)
    adjacency = (
        get_adjacency(phase, threshold)
        if threshold < 0.5 * np.pi
        else np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    )
    weighted_adjacency = normalized_coupling * adjacency

    phase_diff = phase[:, None] - phase[None, :]
    jacobian = adjacency * np.cos(phase_diff)
    np.fill_diagonal(jacobian, -np.dot(weighted_adjacency, np.ones(num_nodes)))

    return jacobian


def solve_kuramoto(
    phase: arr64,
    eval_time: arr64,
    normalized_coupling: float,
    omega: arr64,
    threshold: float,
) -> tuple[arr64, int, float]:
    """
    Solve kuramoto equation using scipy.integrate.solve_ivp
    dphase/dt = omega + sum_j K_ij sin(phase_j - phase_i)

    Args
    phase: [N, ], initial phase of each node
    normalized_coupling: normalized coupling constant
    eval_time: [S+1, ], Evaluation time coordinates
    omega: [N, ], omega (natural angular velocity) of each node

    Returns
    trajectory: [S+1, N, 1], phase at each time step including initial condition
    num_func_evals: Number of function evaluations during solve_ivp
    runtime: Runtime
    """
    start = time.perf_counter()
    result = solve_ivp(
        get_velocity,
        t_span=(eval_time[0], eval_time[-1]),
        y0=phase,
        method="DOP853",
        t_eval=eval_time,
        args=(normalized_coupling, omega, threshold),
        atol=1e-11,  # Around float64 precision
        rtol=1e-11,  # Around float64 precision
    )
    end = time.perf_counter()

    if result.status != 0:
        raise RuntimeError("Error in solve kuramoto equation")

    trajectory = result.y.T  # [S+1, N]
    return trajectory[..., None], result.nfev, end - start


def run(args: argparse.Namespace, rng: np.random.Generator) -> SimulationData:
    # Sample variables for current simulation
    num_nodes = int(
        rng.integers(min(args.num_nodes), max(args.num_nodes), endpoint=True)
    )

    eval_time = sample_eval_time(
        (args.max_time, -1.0), (args.num_steps, -1), args.dt_delta, rng
    )
    phase = sample_phase(num_nodes, rng)
    omega = sample_omega(num_nodes, rng)
    coupling = sample_coupling(args.coupling, rng)

    # Simulate
    trajectory, nfev, runtime = solve_kuramoto(
        phase, eval_time, coupling / num_nodes, omega, args.threshold
    )

    # Store the result
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    nx.set_node_attributes(  # [S+1, 1]
        graph, {node: trajectory[:, node] for node in range(num_nodes)}, "trajectory"
    )
    nx.set_node_attributes(  # [1, ]
        graph,
        {node: o.reshape(1) for node, o in zip(range(num_nodes), omega)},
        "node_attr",
    )
    nx.set_edge_attributes(graph, np.empty(0), "edge_attr")

    return {
        "graph": graph,
        "eval_time": eval_time,  # [S+1, ]
        "glob_attr": np.array([[coupling / num_nodes]]),  # [1, 1]
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
