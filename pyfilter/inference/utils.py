from dataclasses import dataclass

import torch
from pyro.distributions import MultivariateNormal, Normal
from torch.linalg import cholesky_ex

from .qmc import EngineContainer


@dataclass
class MeanChol(object):
    mean: torch.Tensor
    chol: torch.Tensor


class QuasiMultivariateNormal(MultivariateNormal):
    r"""
    Implements a multivariate random normal class which enables sampling by inversion.
    """

    has_rsample = False

    def __init__(
        self, quasi_engine: EngineContainer, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None
    ):
        super().__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)
        self.quasi_engine = quasi_engine

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        probs = self.quasi_engine.sample(sample_shape).to(self.loc.device)

        loc = torch.zeros(shape, device=self.loc.device)
        scale = torch.ones(shape, device=self.loc.device)
        normal = Normal(loc, scale)

        eps = normal.icdf(probs)

        return self.loc + self._unbroadcasted_scale_tril.matmul(eps.unsqueeze(-1)).squeeze(-1)


def calc_mean_chol(x: torch.Tensor, w: torch.Tensor) -> MeanChol:
    r"""
    Calculates the mean and covariance for a multivariate normal density.
    """

    mean = w @ x
    centralized = x - mean
    cov = (w * centralized.t()).matmul(centralized)

    cholesky, info = cholesky_ex(cov)

    if (info > 0).any():
        cholesky = cov.diag().sqrt().diag()

    return MeanChol(mean, cholesky)


def construct_mvn(x: torch.Tensor, w: torch.Tensor, scale=1.0, quasi_engine: EngineContainer = None) -> MultivariateNormal:
    r"""
    Constructs a multivariate normal distribution from weighted samples.

    Args:
        x (torch.Tensor): samples to use for estimating density, of size ``{batch size, dimension of space}``.
        w (torch.Tensor): weights associated with each row of ``x``.
        scale (float, optional): applies a scaling to the Cholesky factorized covariance matrix of the distribution. Defaults to 1.0.
        quasi_engine (EngineContainer, optional): QMC engine to use. Defaults to None.
    """

    mean_chol = calc_mean_chol(x, w)
    scale_tril = scale * mean_chol.chol

    if not quasi_engine:
        return MultivariateNormal(mean_chol.mean, scale_tril=scale_tril)

    return QuasiMultivariateNormal(quasi_engine, mean_chol.mean, scale_tril=scale_tril)
