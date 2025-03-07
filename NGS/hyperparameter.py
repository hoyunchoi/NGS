from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import yaml


@dataclass(slots=True)
class HyperParameter:
    # Data
    dataset: str
    seed: int
    missing: float
    noise: float

    # Model, threshold for kuramoto system
    depth: int
    emb_dim: int
    dropout: float
    threshold: float | None

    # Learning rate scheduler
    lr: float
    lr_max: float
    lr_max_mult: float
    period: int
    period_mult: float
    warmup: int

    # Training settings
    loss: str
    weight_decay: float
    device: str
    epochs: int
    batch_size: int

    def to_yaml(self: Self, file_path: Path | str) -> None:
        with open(file_path, "w") as f:
            yaml.safe_dump(asdict(self), f)

    @classmethod
    def from_yaml(cls: type[Self], file_path: Path | str) -> Self:
        with open(file_path, "r") as f:
            hp: dict[str, Any] = yaml.safe_load(f)

        return cls(**hp)
