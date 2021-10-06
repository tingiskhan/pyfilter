import torch
from typing import Tuple
from ....timeseries import StochasticProcess, NewState
from ....constants import EPS2 as _EPS

_COV_FACTOR = 1 / _EPS


def propagate_sps(
    spx: NewState, spn: torch.Tensor, process: StochasticProcess, temp_params: Tuple[torch.Tensor, ...]
) -> NewState:
    """
    Propagates sigma points via the process dynamics from :math:`t \\rightarrow t+1`.

    Args:
        spx: The sigma points of the state process at time :math:`t`.
        spn: The sigma points of the distribution at time :math:`t+1`.
        process: The process dynamics via which to propagate the sigma points.
        temp_params: The sigma points are tensors with one more dimension than the original process. As such, we allow
            passing reshaped parameters to override ``process`` with.

    Returns:
        Propagated sigma points.
    """

    is_multidimensional = process.n_dim > 0

    if not is_multidimensional:
        spx = spx.copy(None, spx.values.squeeze(-1))
        spn = spn.squeeze(-1)

    res = process.propagate_conditional(spx, u=spn, parameters=temp_params)

    if not is_multidimensional:
        res.values.unsqueeze_(-1)

    return res


def covariance(a: torch.Tensor, b: torch.Tensor, covariance_weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates the covariance via :math:`w_{c} \cdot a \cdot b^T`.

    Args:
        a: The A matrix.
        b: The B matrix.
        covariance_weights: The weights to use for aggregating the covariance.
    """

    cov = a.unsqueeze(-1) * b.unsqueeze(-2)

    return (covariance_weights.view(-1, 1, 1) * cov).sum(-3)


def get_mean_and_cov(sp: torch.Tensor, wm: torch.Tensor, wc: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Calculates the mean and covariance of the sigma points using weights for mean and covariance.

    Args:
        sp: The sigma points to aggregate.
        wm: The weights to use for calculating the mean.
        wc: The weights to use for calculating the covariance.

    Returns:
        Returns the tuple ``(mean, covariance)``.
    """

    x = (wm.unsqueeze(-1) * sp).sum(-2)
    centered = sp - x.unsqueeze(-2)

    return x, covariance(centered, centered, wc)


def get_bad_inds(*covs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Given one or more batched covariance matrices, find the batches that are badly conditioned and return the
    "aggregated" indices.

    Args:
        covs: One more batched covariance matrices, i.e. tensors of size ``(batch size, dimension, dimension)``.

    Returns:
        Indices of badly conditioned covariances.
    """

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
