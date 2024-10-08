import torch
from torch.amp.autocast_mode import autocast

from NGS.data import get_loader
from NGS.experiment import amp_dtype
from traffic.data import TrafficDataset
from traffic.model import TrafficModel


@torch.no_grad()
def rollout(
    model: TrafficModel,
    dataset: TrafficDataset,
    device: torch.device,
    out_windows: list[int],
) -> dict[int, torch.Tensor]:
    """
    Return: dictionary of rollouts for each window size
    Each trajectory is of shape [N, S, W_out]
    """
    model.eval()
    data_loader = get_loader(dataset, shuffle=False, batch_size=1)

    trajectories: dict[int, list[torch.Tensor]] = {window: [] for window in out_windows}
    for batch_data in data_loader:
        batch_data = batch_data.to(device)

        with autocast(device.type, dtype=amp_dtype(device)):
            model.cache(
                batch_data.node_attr, batch_data.edge_attr, batch_data.glob_attr
            )
            state = model(  # [N, W_out]
                state=batch_data.x,  # [N, 5 * W_in]
                dt=batch_data.dts[0],  # [1, 1]
                edge_index=batch_data.edge_index,  # [2, E]
                batch=batch_data.batch,  # [N, ]
                ptr=batch_data.ptr,  # [2,]
            )
            model.empty_cache()

        for window in out_windows:
            trajectories[window].append(state[:, window - 1])  # S * [N, ]

    return {
        window: torch.stack(trajectory, dim=0)  # [S, N]
        for window, trajectory in trajectories.items()
    }
