from torch.distributions import MultivariateNormal
import torch
from torch.linalg import cholesky_ex


def construct_mvn(x: torch.Tensor, w: torch.Tensor, scale=1.0) -> MultivariateNormal:
    """
    Constructs a multivariate normal distribution from weighted samples.

    Args:
        x: samples from which to create the the multivariate normal distribution, should be of size
            ``(batch size, dimension of space)``.
        w: weights associated to each point of ``x``, should therefore be of size ``batch size``.
        scale: applies a scaling to the Cholesky factorized covariance matrix of the distribution.
    """

    mean = w @ x
    centralized = x - mean
    cov = (w * centralized.t()).matmul(centralized)

    cholesky, info = cholesky_ex(cov)

    if (info > 0).any():
        cholesky = cov.diag().sqrt().diag()

    return MultivariateNormal(mean, scale_tril=scale * cholesky)
