from torch.distributions import MultivariateNormal
import warnings
from typing import Callable
from ..filters import BaseFilter, FilterResult
from .state import AlgorithmState
from torch.distributions import Distribution, Independent
import torch
from typing import Tuple


PropConstructor = Callable[[AlgorithmState, BaseFilter, torch.Tensor], Distribution]


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


def run_pmmh(
    filter_: BaseFilter,
    state: FilterResult,
    prop_kernel: Distribution,
    prop_filt,
    y: torch.Tensor,
    size=torch.Size([]),
    **kwargs
) -> Tuple[torch.Tensor, FilterResult, BaseFilter]:
    """
    Runs one iteration of a vectorized Particle Marginal Metropolis hastings.
    """

    rvs = prop_kernel.sample(size)
    prop_filt.ssm.parameters_from_array(rvs, transformed=True)

    new_res = prop_filt.longfilter(y, bar=False, **kwargs)

    diff_logl = new_res.loglikelihood - state.loglikelihood
    diff_prior = prop_filt.ssm.p_prior() - filter_.ssm.p_prior()

    if isinstance(prop_kernel, Independent) and size == torch.Size([]):
        diff_prop = 0.0
    else:
        diff_prop = prop_kernel.log_prob(filter_.ssm.parameters_to_array(transformed=True)) - prop_kernel.log_prob(rvs)

    log_acc_prob = diff_prop + diff_prior + diff_logl
    res: torch.Tensor = torch.empty_like(log_acc_prob).uniform_().log() < log_acc_prob

    return res, new_res, prop_filt