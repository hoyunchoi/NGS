import copy
import time
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.loader as gLoader
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from NGS.data import NGSDataset, get_loader
from NGS.ema import EMA
from NGS.hyperparameter import HyperParameter
from NGS.model import Model
from NGS.scheduler import CosineScheduler
from path import RESULT_DIR


def amp_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def train(
    model: Model,
    data_loader: gLoader.DataLoader,
    optimizer: optim.Optimizer,
    grad_scaler: GradScaler,
    ema: EMA,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()
    mse = torch.tensor(0.0, device=device)
    mae = torch.tensor(0.0, device=device)

    for batch_data in data_loader:
        batch_data = batch_data.to(device)

        with autocast(device.type, amp_dtype(device)):
            model.cache(
                batch_data.node_attr, batch_data.edge_attr, batch_data.glob_attr
            )
            pred_y = model(  # [BN, state_dim]
                state=batch_data.x,  # [BN, state_dim]
                dt=batch_data.dts[0],  # [B, 1]
                edge_index=batch_data.edge_index,  # [2, BE]
                batch=batch_data.batch,  # [BN, ]
                ptr=batch_data.ptr,  # [B+1, ]
            )
            model.empty_cache()

            true_y = batch_data.y[0]  # [BN, state_dim]

            # Do not use missing data for calculating loss
            pred_y = pred_y[~batch_data.is_missing]
            true_y = true_y[~batch_data.is_missing]
            loss = loss_fn(pred_y, true_y)

            with torch.no_grad():
                mse += F.mse_loss(pred_y, true_y)
                mae += F.l1_loss(pred_y, true_y)

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()  # Replacing loss.backward()
        grad_scaler.step(optimizer)  # Replacing optimizer.step()
        grad_scaler.update()
        ema.step()

    return mse / len(data_loader), mae / len(data_loader)


@torch.no_grad()
def validate(
    model: Model, data_loader: gLoader.DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    mse = torch.tensor(0.0, device=device)
    mae = torch.tensor(0.0, device=device)

    for batch_data in data_loader:
        batch_data = batch_data.to(device)

        pred_ys: list[torch.Tensor] = []  # S * [BN, state_dim]
        with autocast(device.type, amp_dtype(device)):
            model.cache(
                batch_data.node_attr, batch_data.edge_attr, batch_data.glob_attr
            )
            state = batch_data.x  # [BN, state_dim]
            for dt in batch_data.dts:
                state = model(
                    state=state,  # [BN, state_dim]
                    dt=dt,  # [B, 1], For traffic forecasting, [B, 0]
                    edge_index=batch_data.edge_index,  # [2, BE]
                    batch=batch_data.batch,  # [BN, ]
                    ptr=batch_data.ptr,  # [B+1, ]
                )
                pred_ys.append(state.clone())
            model.empty_cache()

            pred_y = torch.stack(pred_ys)  # [W, BN, state_dim]
            true_y = batch_data.y  # [W, BN, state_dim]

            mse += F.mse_loss(pred_y, true_y)
            mae += F.l1_loss(pred_y, true_y)

    return mse / len(data_loader), mae / len(data_loader)


@torch.no_grad()
def rollout(
    model: Model, dataset: NGSDataset, device: torch.device
) -> tuple[list[npt.NDArray[np.float32]], list[int], list[float]]:
    """
    Returns
    trajectories: [Ns, S+1, N, state_dim], including initial condition
    nfevs: [Ns, ], number of function evaluations for each simulation
    runtimes: [Ns, ], runtime for each simulation
    """
    model.eval()
    data_loader = get_loader(dataset, shuffle=False, batch_size=1, pin_memory=True)

    # Warmup
    validate(model, data_loader, device)

    # Measure time and store result
    trajectories: list[npt.NDArray[np.float32]] = []
    nfevs: list[int] = []
    runtimes: list[float] = []
    with autocast(device.type, dtype=amp_dtype(device)):
        for batch_data in data_loader:
            torch.cuda.synchronize()
            start = time.perf_counter()

            batch_data = batch_data.to(device)
            model.cache(
                batch_data.node_attr, batch_data.edge_attr, batch_data.glob_attr
            )
            state = batch_data.x
            trajectory: list[torch.Tensor] = [state.clone()]
            for dt in batch_data.dts:
                state = model(
                    state=state,  # [BN, state_dim]
                    dt=dt,  # [B, 1]
                    edge_index=batch_data.edge_index,  # [2, BE]
                    batch=batch_data.batch,  # [BN, ]
                    ptr=batch_data.ptr,  # [B+1, ]
                )
                trajectory.append(state.clone())
            model.empty_cache()

            torch.cuda.synchronize()
            end = time.perf_counter()

            trajectories.append(torch.stack(trajectory).cpu().numpy())
            nfevs.append(len(batch_data.dts))
            runtimes.append(end - start)

    return trajectories, nfevs, runtimes


def run(
    exp_id: str,
    hp: HyperParameter,
    model: Model,
    train_dataset: NGSDataset,
    val_dataset: NGSDataset,
) -> None:
    device = torch.device(hp.device)
    model = model.to(device)

    # Create data loaders
    train_loader = get_loader(train_dataset, shuffle=True, batch_size=hp.batch_size)
    val_loader = get_loader(val_dataset, shuffle=False, batch_size=hp.batch_size)

    # loss function, optimizer, scheduler, gradient scaler, ema
    if hp.loss == "mse":
        loss_fn = F.mse_loss
    elif hp.loss == "mae":
        loss_fn = F.l1_loss
    else:
        raise ValueError(f"Invalid loss function: {hp.loss}")
    optimizer = optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = CosineScheduler(
        optimizer, hp.lr_max, hp.period, hp.warmup, hp.lr_max_mult, hp.period_mult
    )
    grad_scaler = GradScaler()
    ema = EMA(model)

    # Best state dictionaries
    best_val_mae = float("inf")
    model_state_dict = copy.deepcopy(model.state_dict())
    ema_state_dict = copy.deepcopy(ema.state_dict())

    # Empty variables to store train process
    train_mses: list[torch.Tensor] = []
    train_maes: list[torch.Tensor] = []
    val_mses: list[torch.Tensor] = []
    val_maes: list[torch.Tensor] = []

    # --------------- Start training ---------------
    for epoch in range(hp.epochs):
        # Train
        train_mse, train_mae = train(
            model, train_loader, optimizer, grad_scaler, ema, loss_fn, device
        )
        scheduler.step()  # update learning rate scheduler

        # NaN detected in training: abort
        if train_mse.isnan():
            raise RuntimeError("NaN detected in training")

        # Validation
        with ema():
            val_mse, val_mae = validate(model, val_loader, device)

        # Store result of current epoch
        train_mses.append(train_mse)
        train_maes.append(train_mae)
        val_mses.append(val_mse)
        val_maes.append(val_mae)

        # Check best validation error
        if val_mae < best_val_mae:
            print(f"Best at {epoch=}, {train_mae=:.4e}, {val_mae=:.4e}")
            best_val_mae = val_mae.item()
            model_state_dict = copy.deepcopy(model.state_dict())
            ema_state_dict = copy.deepcopy(ema.state_dict())

    # --------------- Finish training ---------------
    result_dir = RESULT_DIR / exp_id
    result_dir.mkdir(parents=True)

    # Save hyperparameter
    hp.to_yaml(result_dir / "hyperparameter.yaml")

    # Save state dictionaries
    best_checkpoint = {"model": model_state_dict, "ema": ema_state_dict}
    torch.save(best_checkpoint, result_dir / "checkpoint.pth")

    # Save loss history
    loss_df = pd.DataFrame.from_dict(
        {
            "train_mse": [mse.item() for mse in train_mses],
            "train_mae": [mae.item() for mae in train_maes],
            "val_mse": [mse.item() for mse in val_mses],
            "val_mae": [mae.item() for mae in val_maes],
        }
    )
    loss_df.to_csv(result_dir / "loss.txt", sep="\t", index=False)
