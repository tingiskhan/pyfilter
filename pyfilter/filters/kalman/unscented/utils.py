import torch
from typing import Tuple
from ....timeseries import StochasticProcess, NewState

_EPS = torch.finfo(torch.float32).eps
_COV_FACTOR = 1 / _EPS


def propagate_sps(spx: NewState, spn: torch.Tensor, process: StochasticProcess, temp_params: Tuple[torch.Tensor, ...]):
    is_multidimensional = process.n_dim > 0

    if not is_multidimensional:
        spx = spx.copy(None, spx.values.squeeze(-1))
        spn = spn.squeeze(-1)

    res = process.propagate_conditional(spx, u=spn, parameters=temp_params)

    if not is_multidimensional:
        res.values.unsqueeze_(-1)

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


def get_bad_inds(*covs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    singular_or_nan = torch.zeros(covs[0].shape[:-2], device=covs[0].device, dtype=torch.bool)
    for cov in covs:
        cov_det = cov.det()
        singular_or_nan |= (
            torch.isnan(cov_det)
            | (cov_det <= _EPS)
            | (cov_det.abs() >= _COV_FACTOR)
            | (cov.diagonal(dim1=-2, dim2=-1) <= 0.0).any(-1)
        )

    return singular_or_nan
