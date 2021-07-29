import torch
from torch.distributions import Distribution
from typing import Tuple
from .proposals import BaseProposal
from ...state import ParticleState
from ...utils import (
    params_to_tensor,
    parameters_and_priors_from_model,
    params_from_tensor,
    sample_model,
    eval_prior_log_prob,
)
from ....filters import BaseFilter, FilterResult
from ....constants import INFTY


def seed(filter_: BaseFilter, y: torch.Tensor, num_seeds: int, num_chains) -> BaseFilter:
    seed_filter = filter_.copy()

    num_samples = num_chains * num_seeds
    seed_filter.set_nparallel(num_samples)

    sample_model(seed_filter.ssm, (num_samples, 1))

    res = seed_filter.longfilter(y, bar=False)

    params = params_to_tensor(seed_filter.ssm)
    log_likelihood = res.loglikelihood

    log_likelihood = log_likelihood.view(-1)
    log_likelihood[~torch.isfinite(log_likelihood)] = -INFTY

    best_ll = log_likelihood.argmax()

    num_params = sum(p.get_numel(constrained=True) for _, p in parameters_and_priors_from_model(filter_.ssm))
    best_params = params[best_ll]

    filter_.set_nparallel(num_chains)

    sample_model(filter_.ssm, (num_chains, 1))
    params_from_tensor(filter_.ssm, best_params.expand(num_chains, num_params), constrained=False)

    return filter_


def run_pmmh(
    filter_: BaseFilter,
    state: ParticleState,
    proposal: BaseProposal,
    prop_kernel: Distribution,
    prop_filter: BaseFilter,
    y: torch.Tensor,
    size=torch.Size([]),
    **kwargs,
) -> Tuple[torch.Tensor, FilterResult, BaseFilter]:
    """
    Runs one iteration of a vectorized Particle Marginal Metropolis hastings.
    """

    rvs = prop_kernel.sample(size)
    params_from_tensor(prop_filter.ssm, rvs, constrained=False)

    new_res = prop_filter.longfilter(y, bar=False, **kwargs)

    diff_logl = new_res.loglikelihood - state.filter_state.loglikelihood
    diff_prior = (eval_prior_log_prob(prop_filter.ssm, False) - eval_prior_log_prob(filter_.ssm, False)).squeeze()

    new_prop_kernel = proposal.build(state.copy(new_res), prop_filter, y)
    diff_prop = new_prop_kernel.log_prob(params_to_tensor(filter_.ssm, constrained=False)) - prop_kernel.log_prob(rvs)

    log_acc_prob = diff_prop + diff_prior + diff_logl
    accepted: torch.Tensor = torch.empty_like(log_acc_prob).uniform_().log() < log_acc_prob

    proposal.exchange(prop_kernel, new_prop_kernel, accepted)

    return accepted, new_res, prop_filter
