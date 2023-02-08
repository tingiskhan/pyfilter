from typing import TypeVar

import torch
from torch.distributions import Distribution

from ....filters import BaseFilter
from ...context import InferenceContext
from ...state import FilterAlgorithmState
from .proposals import BaseProposal

TFilter = TypeVar("TFilter", bound=BaseFilter)


def run_pmmh(
    context: InferenceContext,
    state: FilterAlgorithmState,
    proposal: BaseProposal,
    proposal_kernel: Distribution,
    proposal_filter: BaseFilter,
    proposal_context: InferenceContext,
    y: torch.Tensor,
    size=torch.Size([]),
    mutate_kernel=False,
) -> torch.BoolTensor:
    r"""
    Runs one iteration of the PMMH update step in which we sample a candidate :math:`\theta^*` from the proposal
    kernel, run a filter for the considered dataset with :math:`\theta^*`, and accept based on the acceptance
    probability given by the article.

    Args:
        context (InferenceContext): parameter context of the main algorithm.
        state (FilterAlgorithmState): latest algorithm state.
        proposal (BaseProposal): proposal to use when building the the candidate kernel :math:`\theta^*`.
        proposal_kernel (Distribution): kernel from which to draw the candidate sample :math:`\theta^*`.
        proposal_filter (BaseFilter): proposal filter to use.
        proposal_context (InferenceContext): parameter context of the proposal filter.
        y (torch.Tensor): see :class:`pyfilter.inference.base.BaseAlgorithm`.
        size (_type_, optional): size of sample to draw from ``proposal_kernel``. Defaults to torch.Size([]).
        mutate_kernel (bool, optional): whether to mutate ``proposal_kernel`` with newly accepted sample. Defaults to False.

    Returns:
        torch.BoolTensor: indices of accepted particles.
    """

    constrained = False

    # Sample parameters and override context
    rvs = proposal_kernel.sample(size)
    proposal_context.unstack_parameters(rvs, constrained=constrained)

    with proposal_context.no_prior_verification():
        proposal_filter.initialize_model(proposal_context)

    # Run proposal filter
    new_res = proposal_filter.batch_filter(y, bar=False)

    # Compare likelihood, prior and proposal kernel
    diff_logl = new_res.loglikelihood - state.filter_state.loglikelihood
    diff_prior = proposal_context.eval_priors(constrained=constrained) - context.eval_priors(constrained=constrained)

    new_prop_kernel = proposal.build(proposal_context, state.replicate(new_res), proposal_filter, y)
    params_as_tensor = context.stack_parameters(constrained=constrained)

    diff_prop = new_prop_kernel.log_prob(params_as_tensor) - proposal_kernel.log_prob(rvs)

    # Generate acceptance probabilities
    log_acc_prob = diff_prop + diff_prior + diff_logl
    accepted: torch.BoolTensor = torch.empty_like(log_acc_prob).uniform_().log() < log_acc_prob

    # Exchange filters and states
    state.filter_state.exchange(new_res, accepted)
    context.exchange(proposal_context, accepted)

    if mutate_kernel:
        proposal.exchange(proposal_kernel, new_prop_kernel, accepted)

    return accepted
