import torch
from torch.distributions import MultivariateNormal
import warnings


def _construct_mvn(x: torch.Tensor, w: torch.Tensor, scale=1.0):
    """
    Constructs a multivariate normal distribution of weighted samples.
    """

    mean = (x * w.unsqueeze(-1)).sum(0)
    centralized = x - mean
    cov = torch.matmul(w * centralized.t(), centralized)

    if cov.det() == 0.0:
        chol = cov.diag().sqrt().diag()
    else:
        chol = cov.cholesky()

    return MultivariateNormal(mean, scale_tril=scale * chol)


def experimental(func):
    def wrapper(obj, *args, **kwargs):
        warnings.warn(f"{obj:s} is an experimental algorithm, use at own risk")

        return func(obj, *args, **kwargs)

    return wrapper


def preliminary(func):
    def wrapper(obj, *args, **kwargs):
        warnings.warn(f"{obj:s} is only a preliminary version algorithm, use at own risk")

        return func(obj, *args, **kwargs)

    return wrapper
