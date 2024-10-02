import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))
from NGS.data import NGSDataset, add_missing, add_noise, preprocess
from NGS.experiment import run
from NGS.hyperparameter import HyperParameter
from path import DATA_DIR
from rossler.model import RosslerModel


def main() -> None:
    # Random experiment ID
    exp_id = "".join(np.random.choice(list(string.ascii_lowercase + string.digits), 8))

    hp = HyperParameter(
        dataset="rossler_train",
        seed=42,
        missing=0.1,
        noise=0.001,
        emb_dim=32,
        depth=2,
        dropout=0.0,
        threshold=None,
        lr=1e-5,
        lr_max=1e-3,
        period=200,
        warmup=20,
        lr_max_mult=0.5,
        period_mult=1.0,
        loss="mse",
        device="cuda:0",
        epochs=1540,
        batch_size=16,
    )

    df = pd.read_pickle(DATA_DIR / f"{hp.dataset}.pkl")
    train, val = preprocess(df, val_ratio=0.2)

    # Incomplete data
    rng = np.random.default_rng(hp.seed)
    seed_missing, seed_noise = rng.integers(42, size=(2,))
    add_missing(train, hp.missing, seed_missing)
    add_noise(train, hp.noise, seed_noise)

    # Dataset for train, validation
    train_dataset = NGSDataset(**train, window=1)
    val_dataset = NGSDataset(**val, window=-1)

    # Model
    model = RosslerModel(hp.emb_dim, hp.depth, hp.dropout)

    print(f"Start running {exp_id=}")
    run(exp_id, hp, model, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
