import os
import string
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))
from NGS.experiment import run
from NGS.hyperparameter import HyperParameter
from path import DATA_DIR
from traffic.data import TrafficDataset, preprocess
from traffic.model import TrafficModel


def main() -> None:
    # Random experiment ID
    exp_id = "".join(np.random.choice(list(string.ascii_lowercase + string.digits), 8))

    hp = HyperParameter(
        dataset="PEMS-BAY",
        seed=42,
        missing=0.0,
        noise=0.0,
        emb_dim=64,
        depth=2,
        dropout=0.2,
        threshold=None,
        lr=1e-5,
        lr_max=8e-4,
        period=50,
        warmup=5,
        lr_max_mult=0.5,
        period_mult=1.0,
        loss="mae",
        weight_decay=0.5,
        device="cuda",
        epochs=275,
        batch_size=16,
    )

    df = cast(pd.DataFrame, pd.read_hdf(DATA_DIR / "PEMS-BAY/pems-bay.h5"))
    *_, adj = pd.read_pickle(DATA_DIR / "PEMS-BAY/adj_mx_bay.pkl")
    preprocessed = preprocess(df, adj)

    trajectory = preprocessed["trajectory"]
    time_in_day = preprocessed["time_in_day"]
    day_in_week = preprocessed["day_in_week"]
    edge_indices = preprocessed["edge_index"]
    edge_attrs = preprocessed["edge_attr"]
    indicies_train = preprocessed["indicies_train"]
    indicies_val = preprocessed["indicies_val"]

    # Dataset for train, validation
    train_dataset = TrafficDataset(trajectory, time_in_day, day_in_week, indicies_train, edge_indices, edge_attrs)
    val_dataset = TrafficDataset(trajectory, time_in_day, day_in_week, indicies_val, edge_indices, edge_attrs)

    # Model
    num_nodes = trajectory.shape[1]
    model = TrafficModel(num_nodes, hp.emb_dim, hp.depth, hp.dropout)

    print(f"Start running {exp_id=}")
    run(exp_id, hp, model, train_dataset, val_dataset)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
