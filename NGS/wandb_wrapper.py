import string
from typing import Any, cast

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from path import WANDB_DIR


class DummyWandbRun:
    def log(self, _: dict[str, Any]) -> None:
        return

    @property
    def summary(self) -> dict[str, Any]:
        return {}

    def finish(self) -> None:
        return

    @property
    def name(self) -> str:
        return "".join(
            np.random.choice(list(string.ascii_lowercase + string.digits), 8)
        )


def config_wandb(use_wandb: bool, hp_dict: dict[str, Any]) -> WandbRun | DummyWandbRun:
    if use_wandb:
        wandb_run = cast(
            WandbRun,
            wandb.init(dir=WANDB_DIR, config=hp_dict, project=hp_dict["wandb"]),
        )
        wandb_run.name = wandb_run.id
    else:
        wandb_run = DummyWandbRun()
    return wandb_run
