from ....filters import BaseFilter, FilterResult
from torch.distributions import Distribution, Independent
import torch
from typing import Tuple
from ....constants import INFTY
from ...state import AlgorithmState
from typing import Callable


PropConstructor = Callable[[AlgorithmState, BaseFilter, torch.Tensor], Distribution]


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


def seed(filter_: BaseFilter, y: torch.Tensor, num_seeds: int, num_chains) -> BaseFilter:
    # ===== Construct and run filter ===== #
    seed_filter = filter_.copy()

    num_samples = num_chains * num_seeds
    seed_filter.set_nparallel(num_samples)
    seed_filter.ssm.sample_params(num_samples)
    seed_filter.ssm.viewify_params((num_samples, 1))

    res = seed_filter.longfilter(y, bar=False)

    # ===== Find best parameters ===== #
    params = seed_filter.ssm.parameters_to_array()
    log_likelihood = res.loglikelihood

    log_likelihood = log_likelihood.view(-1)
    log_likelihood[~torch.isfinite(log_likelihood)] = -INFTY

    best_ll = log_likelihood.argmax()

    num_params = sum(p.numel_() for p in filter_.ssm.trainable_parameters)
    best_params = params[best_ll]

    # ===== Set filter parameters ===== #
    filter_.set_nparallel(num_chains)
    filter_.ssm.sample_params(num_chains)
    filter_.ssm.viewify_params((num_chains, 1))
    filter_.ssm.parameters_from_array(best_params.expand(num_chains, num_params))

    return filter_
