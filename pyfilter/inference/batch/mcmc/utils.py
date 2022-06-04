import torch
from torch.distributions import Distribution
from typing import TypeVar
from .proposals import BaseProposal
from ...context import ParameterContext
from ...state import FilterAlgorithmState
from ....filters import BaseFilter


TFilter = TypeVar("TFilter", bound=BaseFilter)


def run_pmmh(
    context: ParameterContext,
    state: FilterAlgorithmState,
    proposal: BaseProposal,
    proposal_kernel: Distribution,
    proposal_filter: BaseFilter,
    proposal_context: ParameterContext,
    y: torch.Tensor,
    size=torch.Size([]),
    mutate_kernel=False,
) -> torch.BoolTensor:
    """
    Runs one iteration of the PMMH update step in which we sample a candidate :math:`\\theta^*` from the proposal
    kernel, run a filter for the considered dataset with :math:`\\theta^*`, and accept based on the acceptance
    probability given by the article. If accepted, we call the ``.exchange(...)`` method of ``filter_`` and ``state``.

    Args:
        context: the parameter context of the main algorithm.
        state: the latest algorithm state.
        proposal: the proposal to use when generating the candidate sample :math:`\\theta^*`.
        proposal_kernel: the kernel from which to draw the candidate sample :math:`\\theta^*`. To clarify, ``proposal``
            corresponds to the ``BaseProposal`` class that was used when generating ``prop_kernel``.
        proposal_filter: the proposal filter to use.
        proposal_context: the parameter context of the proposal filter.
        y: see ``pyfilter.inference.base.BaseAlgorithm``.
        size: optional parameter specifying the number of num_samples to draw from ``proposal_kernel``. Should be empty
            if we draw from an independent kernel.
        mutate_kernel: optional parameter specifying whether to update ``proposal_kernel`` with the newly accepted
            candidate sample.

    Returns:
        Returns the candidate sample(s) that were accepted.
    """

    constrained = False

    rvs = proposal_kernel.sample(size)
    proposal_context.unstack_parameters(rvs, constrained=constrained)

    new_res = proposal_filter.batch_filter(y, bar=False)

    diff_logl = new_res.loglikelihood - state.filter_state.loglikelihood
    diff_prior = proposal_context.eval_priors(constrained=constrained) - context.eval_priors(constrained=constrained)

    new_prop_kernel = proposal.build(proposal_context, state.replicate(new_res), proposal_filter, y)
    params_as_tensor = context.stack_parameters(constrained=constrained)

    diff_prop = new_prop_kernel.log_prob(params_as_tensor) - proposal_kernel.log_prob(rvs)

    log_acc_prob = diff_prop + diff_prior.squeeze(-1) + diff_logl
    accepted: torch.BoolTensor = torch.empty_like(log_acc_prob).uniform_().log() < log_acc_prob

    state.filter_state.exchange(new_res, accepted)

    broadcasted_mask = accepted.unsqueeze(-1)
    context.exchange(proposal_context, broadcasted_mask)

    if mutate_kernel:
        proposal.exchange(proposal_kernel, new_prop_kernel, broadcasted_mask)

    return accepted
