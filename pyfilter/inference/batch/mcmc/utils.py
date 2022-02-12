import torch
from torch.distributions import Distribution
from typing import TypeVar
from .proposals import BaseProposal
from ...state import FilterAlgorithmState
from ....filters import BaseFilter, ParticleFilter
from ....constants import INFTY

TFilter = TypeVar("TFilter", bound=BaseFilter)


def seed(filter_: TFilter, y: torch.Tensor, num_seeds: int, num_chains: int) -> TFilter:
    """
    Seeds the initial sample(s) of the Markov chain by running the chosen filter on a subset of the data, and then
    picking the best one in terms of total log likelihood, i.e. :math:`p_\\theta(y_{1:t}) \\cdot p(\\theta)`.

    Args:
        filter_: The filter to use in the PMCMC algorithm.
        y: The subset of the data, of shape ``(number of observations, [dimension of observation space])``.
        num_seeds: The number of seeds to consider. The number of total samples to pick from is given by
            ``num_seeds * num_chains``.
        num_chains: The number of chains to consider in the base PMCMC algorithm.

    Returns:
        Returns ``filter_`` with the best parameters.
    """

    num_samples = num_chains * num_seeds
    filter_.set_num_parallel(num_samples)

    size = torch.Size([num_samples, 1] if isinstance(filter_, ParticleFilter) else [num_samples])
    filter_.ssm.sample_params(size)

    res = filter_.longfilter(y, bar=False)

    params = filter_.ssm.concat_parameters(constrained=True)
    log_likelihood = res.loglikelihood + filter_.ssm.eval_prior_log_prob(constrained=True).squeeze()
    log_likelihood[~torch.isfinite(log_likelihood)] = -INFTY

    return params[log_likelihood.argmax()]


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
    proposal_filter.ssm.update_parameters_from_tensor(rvs, constrained=False)

    new_res = proposal_filter.longfilter(y, bar=False, **kwargs)

    diff_logl = new_res.loglikelihood - state.filter_state.loglikelihood
    diff_prior = (proposal_filter.ssm.eval_prior_log_prob(False) - filter_.ssm.eval_prior_log_prob(False)).squeeze()

    new_prop_kernel = proposal.build(state.replicate(new_res), proposal_filter, y)
    params_as_tensor = filter_.ssm.concat_parameters(constrained=False, flatten=True)

    diff_prop = new_prop_kernel.log_prob(params_as_tensor) - proposal_kernel.log_prob(rvs)

    log_acc_prob = diff_prop + diff_prior + diff_logl
    accepted: torch.Tensor = torch.empty_like(log_acc_prob).uniform_().log() < log_acc_prob

    state.filter_state.exchange(new_res, accepted)
    filter_.exchange(proposal_filter, accepted)

    if mutate_kernel:
        proposal.exchange(proposal_kernel, new_prop_kernel, accepted)

    return accepted
