from ...filters import BaseFilter, FilterResult
from torch.distributions import Distribution
import torch
from typing import Tuple


def run_pmmh(
    filter_: BaseFilter, state: FilterResult, prop_kernel: Distribution, prop_filt, y: torch.Tensor
) -> Tuple[torch.Tensor, FilterResult, BaseFilter]:
    """
    Runs one iteration of a vectorized Particle Marginal Metropolis hastings.
    """

    rvs = prop_kernel.sample(filter_._n_parallel)
    prop_filt.ssm.parameters_from_array(rvs, transformed=True)

    new_res = prop_filt.longfilter(y, bar=False)

    diff_logl = new_res.loglikelihood - state.loglikelihood
    diff_prior = prop_filt.ssm.p_prior() - filter_.ssm.p_prior()
    diff_prop = prop_kernel.log_prob(filter_.ssm.parameters_to_array(transformed=True)) - prop_kernel.log_prob(rvs)

    log_acc_prob = diff_prop + diff_prior + diff_logl
    res: torch.Tensor = torch.empty_like(log_acc_prob).uniform_().log() < log_acc_prob

    return res, new_res, prop_filt
