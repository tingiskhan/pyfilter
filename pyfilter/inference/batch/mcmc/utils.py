import torch
from torch.distributions import Distribution
from typing import TypeVar
from .proposals import BaseProposal
from ...state import FilterAlgorithmState
from ...utils import (
    params_to_tensor,
    parameters_and_priors_from_model,
    params_from_tensor,
    sample_model,
    eval_prior_log_prob,
)
from ....filters import BaseFilter
from ....constants import INFTY

TFilter = TypeVar("TFilter", bound=BaseFilter)


def seed(filter_: TFilter, y: torch.Tensor, num_seeds: int, num_chains: int) -> TFilter:
    """
    Seeds the initial sample(s) of the Markov chain by running the chosen filter on a subset of the data, and then
    picking the best one in terms of total log likelihood, i.e. :math:`p_\\theta(y_{1:t}) \cdot p(\\theta)`.

    Args:
        filter_: The filter to use in the PMCMC algorithm.
        y: The subset of the data, of shape ``(number of observations, [dimension of observation space])``.
        num_seeds: The number of seeds to consider. The number of total samples to pick from is given by
            ``num_seeds * num_chains``.
        num_chains: The number of chains to consider in the base PMCMC algorithm.

    Returns:
        Returns ``filter_`` with the best parameters.
    """

    seed_filter = filter_.copy()

    num_samples = num_chains * num_seeds
    seed_filter.set_num_parallel(num_samples)

    sample_model(seed_filter.ssm, (num_samples, 1))

    res = seed_filter.longfilter(y, bar=False)

    params = params_to_tensor(seed_filter.ssm)
    log_likelihood = res.loglikelihood + eval_prior_log_prob(seed_filter.ssm).squeeze()

    log_likelihood = log_likelihood.view(-1)
    log_likelihood[~torch.isfinite(log_likelihood)] = -INFTY

    best_ll = log_likelihood.argmax()

    num_params = sum(p.get_numel(constrained=True) for _, p in parameters_and_priors_from_model(filter_.ssm))
    best_params = params[best_ll]

    filter_.set_num_parallel(num_chains)

    sample_model(filter_.ssm, (num_chains, 1))
    params_from_tensor(filter_.ssm, best_params.expand(num_chains, num_params), constrained=False)

    return filter_


def run_pmmh(
    filter_: TFilter,
    state: FilterAlgorithmState,
    proposal: BaseProposal,
    proposal_kernel: Distribution,
    y: torch.Tensor,
    size=torch.Size([]),
    mutate_kernel=False,
    **kwargs,
) -> torch.Tensor:
    """
    Runs one iteration of the PMMH update step in which we sample a candidate :math:`\\theta^*` from the proposal
    kernel, run a filter for the considered dataset with :math:`\\theta^*`, and accept based on the acceptance
    probability given by the article. If accepted, we call the ``.exchange(...)`` method of ``filter_`` and ``state``.

    Args:
        filter_: The filter with the latest accepted candidate samples.
        state: The latest algorithm state.
        proposal: The proposal to use when generating the candidate sample :math:`\\theta^*`.
        proposal_kernel: The kernel from which to draw the candidate sample :math:`\\theta^*`. To clarify, ``proposal``
            corresponds to the ``BaseProposal`` class that was used when generating ``prop_kernel``.
        y: See ``pyfilter.inference.base.BaseAlgorithm``.
        size: Optional parameter specifying the number of samples to draw from ``proposal_kernel``. Should be empty if
            we draw from an independent kernel.
        mutate_kernel: Optional parameter specifying whether to update ``proposal_kernel`` with the newly accepted
            candidate sample.
        kwargs: Kwargs passed

    Returns:
        Returns the candidate sample(s) that were accepted.
    """

    proposal_filter = filter_.copy()

    rvs = proposal_kernel.sample(size)
    params_from_tensor(proposal_filter.ssm, rvs, constrained=False)

    new_res = proposal_filter.longfilter(y, bar=False, **kwargs)

    diff_logl = new_res.loglikelihood - state.filter_state.loglikelihood
    diff_prior = (eval_prior_log_prob(proposal_filter.ssm, False) - eval_prior_log_prob(filter_.ssm, False)).squeeze()

    new_prop_kernel = proposal.build(state.replicate(new_res), proposal_filter, y)
    diff_prop = new_prop_kernel.log_prob(params_to_tensor(filter_.ssm, constrained=False)) - proposal_kernel.log_prob(
        rvs
    )

    log_acc_prob = diff_prop + diff_prior + diff_logl
    accepted: torch.Tensor = torch.empty_like(log_acc_prob).uniform_().log() < log_acc_prob

    state.filter_state.exchange(new_res, accepted)
    filter_.exchange(proposal_filter, accepted)

    if mutate_kernel:
        proposal.exchange(proposal_kernel, new_prop_kernel, accepted)

    return accepted
