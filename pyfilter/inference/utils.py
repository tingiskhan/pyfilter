from dataclasses import dataclass

from pyro.distributions import MultivariateNormal, Normal
import torch
from torch.linalg import cholesky_ex

from .qmc import QuasiRegistry


@dataclass
class MeanChol(object):
    mean: torch.Tensor
    chol: torch.Tensor


class QuasiMultivariateNormal(MultivariateNormal):
    r"""
    Implements a multivariate random normal class which enables sampling by inversion.
    """

    has_rsample = False

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)

        # TODO: This should be in __init__...
        QuasiRegistry.add_engine(self.event_shape.numel())
        probs = QuasiRegistry.sample(self.event_shape.numel(), sample_shape).to(self.loc.device)

        probs.resize_(shape)

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


def construct_mvn(x: torch.Tensor, w: torch.Tensor, scale=1.0, quasi: bool = False) -> MultivariateNormal:
    """
    Constructs a multivariate normal distribution from weighted samples.

    Args:
        x: samples from which to create the the multivariate normal distribution, should be of size
            ``(batch size, dimension of space)``.
        w: weights associated to each point of ``x``, should therefore be of size ``batch size``.
        scale: applies a scaling to the Cholesky factorized covariance matrix of the distribution.
        quasi: whether to use quasi random sampling.
    """

    mean_chol = calc_mean_chol(x, w)
    scale_tril = scale * mean_chol.chol

    if not quasi:
        return MultivariateNormal(mean_chol.mean, scale_tril=scale_tril)

    return QuasiMultivariateNormal(mean_chol.mean, scale_tril=scale_tril)
