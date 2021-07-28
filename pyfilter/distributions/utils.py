from torch.distributions import MultivariateNormal
import torch


def construct_mvn(x: torch.Tensor, w: torch.Tensor, scale=1.0):
    """
    Constructs a multivariate normal distribution from weighted samples.
    """

    mean = (x * w.unsqueeze(-1)).sum(0)
    centralized = x - mean
    cov = torch.matmul(w * centralized.t(), centralized)

    if cov.det() == 0.0:
        chol = cov.diag().sqrt().diag()
    else:
        chol = cov.cholesky()

    return MultivariateNormal(mean, scale_tril=scale * chol)
