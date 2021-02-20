import torch
from typing import Tuple
from ....timeseries import StochasticProcess, TimeseriesState


def propagate_sps(
    spx: TimeseriesState, spn: torch.Tensor, process: StochasticProcess, temp_params: Tuple[torch.Tensor, ...]
):
    is_multidimensional = process.n_dim > 0

    if not is_multidimensional:
        spx = spx.copy(spx.state.squeeze(-1))
        spn = spn.squeeze(-1)

    res = process.propagate_conditional(spx, u=spn, parameters=temp_params)

    if not is_multidimensional:
        res.state.unsqueeze_(-1)

    return res


def covariance(a: torch.Tensor, b: torch.Tensor, wc: torch.Tensor):
    """
    Calculates the covariance from a * b^t
    """
    cov = a.unsqueeze(-1) * b.unsqueeze(-2)

    return (wc[:, None, None] * cov).sum(-3)


def get_meancov(spxy: torch.Tensor, wm: torch.Tensor, wc: torch.Tensor):
    x = (wm.unsqueeze(-1) * spxy).sum(-2)
    centered = spxy - x.unsqueeze(-2)

    return x, covariance(centered, centered, wc)
