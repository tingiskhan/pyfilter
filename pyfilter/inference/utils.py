from torch.distributions import MultivariateNormal
import torch

if torch.__version__ >= "1.9.0":
    chol_fun = torch.linalg.cholesky
else:
    chol_fun = torch.cholesky


def construct_mvn(x: torch.Tensor, w: torch.Tensor, scale=1.0) -> MultivariateNormal:
    """
    Constructs a multivariate normal distribution from weighted samples.

    Args:
        x: samples from which to create the the multivariate normal distribution, should be of size
            ``(batch size, dimension of space)``.
        w: weights associated to each point of ``x``, should therefore be of size ``batch size``.
        scale: applies a scaling to the Cholesky factorized covariance matrix of the distribution.
    """

    mean = (x * w.unsqueeze(-1)).sum(0)
    centralized = x - mean
    cov = torch.matmul(w * centralized.t(), centralized)

    if cov.det() <= 0.0:
        chol = cov.diag().sqrt().diag()
    else:
        chol = chol_fun(cov)

    return MultivariateNormal(mean, scale_tril=scale * chol)
