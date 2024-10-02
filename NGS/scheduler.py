import math
from typing import Self

from torch import optim
from torch.optim.lr_scheduler import LRScheduler


class CosineScheduler(LRScheduler):
    """
    Extended version of pytorch schedular: CosineAnnealingWarmRestarts
    learning rate (lr) evolves as the following equation

    ----------------------------------------------------------------------------
    During warmup stage:
    lr = lr_min + (lr_max - lr_min)/warmup * t

    During cosine stage:
    lr = lr_min + 0.5 * (lr_max - lr_min) * [1 + cos (t-warmup)/T * pi]
    ----------------------------------------------------------------------------
    Note: 1 cycle = warmup stage + cosine stage

    lr_min: Minimum learning rate. Extracted from initial learning rate of optimizer
    lr_max: Maximum learning rate
    t: Number of epochs since current cycle starts.
    warmup: Number of epochs for warmup. If 0, skip warmup stage.
    T(period): Number of epochs for cosine stage.
    lr_max_mult: Decreasing rate of lr_max per each cycle
    period_mult: Increasing rate of period per each cycle
    """

    # # Variables defined at LRScheduler
    # optimizer: optim.Optimizer
    # last_epoch: int
    # base_lrs: list[float]

    def __init__(
        self,
        optimizer: optim.Optimizer,
        lr_max: float,
        period: int,
        warmup: int,
        lr_max_mult: float,
        period_mult: float,
        last_epoch: int = -1,
    ) -> None:
        # Check input
        assert period > 0, f"At cosine schedular, {period=}"
        assert lr_max > 0, f"At cosine schedular, {lr_max=}"
        assert warmup >= 0, f"At cosine schedular, {warmup=}"
        assert period_mult >= 1, f"At cosine schedular, {period_mult=}"
        assert lr_max_mult > 0, f"At cosine schedular, {lr_max_mult=}"

        # Constant properties of schedular
        self._warmup = int(warmup)
        self._lr_max_mult = float(lr_max_mult)
        self._period_mult = float(period_mult)

        # Properties varying per cycle
        self._cycle_start = 0  # Epoch when current cycle is started
        self._period = period  # Period of cosine stage at current cycle
        self._lr_max = lr_max  # lr_max used for current cycle

        # Current location of schedular since cycle start
        # Since super().__init__ calls self.step once, it should be -1 at first
        self.__t = -1

        super().__init__(optimizer, last_epoch)

    def step(self, epoch: int | float | None = None) -> None:
        """
        epoch: epoch + iteration/(tot iteration per epoch)
        If Nothing is given, increase last_epoch
        """
        if epoch is None:
            # Input epoch is None: step up 1 epoch
            self.__t += 1
            self.last_epoch += 1
        else:
            # Input epoch is given
            self.__t = epoch - self._cycle_start
            self.last_epoch = math.floor(epoch)

        # If current location is bigger than period of cycle, update cycle
        if self.__t >= self._warmup + self._period:
            self._update_cycle()

        # Set learning rate of optimizer
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group["lr"] = lr

    def _get_lr(self) -> list[float]:
        def warmup_lr(lr_min: float, lr_max: float, warmup: int, t: float) -> float:
            """lr_min + (lr_max - lr_min)/warmup * t"""
            return lr_min + max(lr_max - lr_min, 0.0) / warmup * t

        def get_cosine_lr(
            lr_min: float, lr_max: float, warmup: int, t: float, period: int
        ) -> float:
            """lr_min + 0.5 * (lr_max - lr_min) * [1 + cos (t-warmup)/T * pi]"""
            cosine_term = math.cos((t - warmup) / period * math.pi)
            return lr_min + 0.5 * max(lr_max - lr_min, 0.0) * (1.0 + cosine_term)

        # Warmup stage
        if self.__t < self._warmup:
            return [
                warmup_lr(base_lr, self._lr_max, self._warmup, self.__t)
                for base_lr in self.base_lrs
            ]

        # Cosine Stage
        return [
            get_cosine_lr(base_lr, self._lr_max, self._warmup, self.__t, self._period)
            for base_lr in self.base_lrs
        ]

    def _update_cycle(self) -> None:
        # Start of current cycle is now increased by period of cycle
        self._cycle_start += self._warmup + self._period

        # Reset current location of new cycle
        self.__t -= self._warmup + self._period

        # Period of cycle is updated by multiplier
        self._period = int(self._period * self._period_mult)

        # Maximum learning rate is updated by multipler
        self._lr_max *= self._lr_max_mult

    def resume(self, epoch: int) -> None:
        """load scheduler based on input epoch"""
        if epoch > 0:
            for _ in range(epoch):
                self.step()

        # Set learning rate of optimizer
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group["lr"] = lr

    def get_lr(
        self: Self, max_epoch: int, num_batch: int = 0
    ) -> tuple[list[float], list[float]]:
        optimizer = self.optimizer

        epochs: list[float] = []
        lrs: list[float] = []

        if num_batch == 0:
            for epoch in range(max_epoch):
                epochs.append(epoch)
                lrs.append(optimizer.param_groups[0]["lr"])
                self.step()
            return epochs, lrs

        for epoch in range(max_epoch):
            for batch in range(num_batch):
                epochs.append(epoch + batch / num_batch)
                lrs.append(optimizer.param_groups[0]["lr"])
                self.step(epoch + batch / num_batch)
        return epochs, lrs

    def get_updated_epochs(self: Self, max_epoch: int) -> list[int]:
        updated_epoch = [0]
        idx = 0
        while updated_epoch[-1] <= max_epoch:
            period = int(self._period * self._period_mult**idx)
            updated_epoch.append(updated_epoch[-1] + period + self._warmup)
            idx += 1
        return updated_epoch[1:-1]
