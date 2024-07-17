from __future__ import annotations

import contextlib
import copy
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        min_decay: float = 0.0,
        warmup: int = 1000,
        gamma: float = 1.0,
        power: float = 2.0 / 3.0,
    ) -> None:
        """
        model: The model to apply EMA. This class will hold same number of parameters
        decay factor of the EMA is given as follows

            decay = clip(1 - (gamma * step)^(-power), min=min_decay, max=decay)

        Here, step is (number of call of self.step - warmup)
        """
        self.parameters = list(model.parameters())  # Reference to original parameters
        self.ema_parameters = [param.detach().clone() for param in self.parameters]

        # Base decay factor
        self.max_decay = decay

        # EMA schedule
        self.min_decay = min_decay
        self.warmup = warmup
        self.gamma = gamma
        self.power = power

        # Current state
        self.num_steps = 0  # Number of step calls
        self.backup: list[torch.Tensor] = []  # Temporary backup parameters

    def compute_decay(self, step: int) -> float:
        """Compute decay factor at given step"""
        # Warmup
        step -= self.warmup
        if step <= 0:
            return 0.0

        # Compute decay factor according to schedule
        decay = 1.0 - (self.gamma * step) ** -self.power

        # Clip current decay factor between [self.min_decay, self.max_decay]
        return min(max(decay, self.min_decay), self.max_decay)

    @torch.no_grad()
    def step(self) -> None:
        """
        Update EMA parameters
        Note that EMA parameters does not include gradient
        """
        self.num_steps += 1
        decay = self.compute_decay(self.num_steps)

        for ema_param, param in zip(self.ema_parameters, self.parameters):
            if param.requires_grad:
                ema_param -= (1.0 - decay) * (ema_param - param)
            else:
                ema_param = param.clone()

    @torch.no_grad()
    def store(self) -> None:
        """Store input parameters to backup"""
        self.backup = [param.clone() for param in self.parameters]

    def copy_to(self) -> None:
        """Copy EMA parameters to input parameters"""
        for ema_param, param in zip(self.ema_parameters, self.parameters):
            param.data.copy_(ema_param.data)

    def restore(self) -> None:
        """Restore backup to the input parameters"""
        if not self.backup:
            raise RuntimeError("This EMA has no `store()`ed weights to `restore()`")
        for backup_param, param in zip(self.backup, self.parameters):
            param.data.copy_(backup_param.data)
        self.backup = []

    @contextlib.contextmanager
    def __call__(self):
        """
        Context manager for applying EMA parameters to the model.
        """
        self.store()
        self.copy_to()
        try:
            yield
        finally:
            self.restore()

    def all_reduce(self) -> None:
        """
        Used for DistributedDataParallel, all reduce the ema parameters
        """
        for ema_param in self.ema_parameters:
            dist.all_reduce(ema_param.data, op=dist.ReduceOp.AVG)

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_steps": self.num_steps,
            "ema_parameters": self.ema_parameters,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.num_steps = state_dict["num_steps"]
        self.ema_parameters = state_dict["ema_parameters"]
