from .base import BaseKernel
from ...utils import _construct_mvn
import torch
from torch.distributions import Distribution
from ...batch.mcmc.utils import run_pmmh, PropConstructor


class SymmetricMH(object):
    def __call__(self, state, filter_, y):
        values = filter_.ssm.parameters_to_array(transformed=True)
        weights = state.normalized_weights()

        return _construct_mvn(values, weights, scale=1.1)  # Same scale in in particles


class ParticleMetropolisHastings(BaseKernel):
    def __init__(self, n_steps=1, proposal: PropConstructor = None, **kwargs):
        """
        Implements a base class for the particle Metropolis Hastings class.
        :param n_steps: The number of steps to perform
        """

        super().__init__(**kwargs)

        self._n_steps = n_steps
        self._proposal = proposal
        self.accepted = None or SymmetricMH()

    def define_pdf(self, values: torch.Tensor, weights: torch.Tensor, inds: torch.Tensor) -> Distribution:
        """
        The method to be overridden by the user for defining the kernel to propagate the parameters. Note that the
        parameters should be propagated in transformed space.
        """

        raise NotImplementedError()

    def _update(self, filter_, state, y, *args):
        prop_filt = filter_.copy((*filter_.n_parallel, 1))

        for _ in range(self._n_steps):
            # ===== Find the best particles ===== #
            inds = self._resampler(state.normalized_weights(), normalized=True)

            # ===== Construct distribution ===== #
            dist = self._proposal(state, filter_, y)

            # ===== Choose particles ===== #
            filter_.resample(inds)
            state.filter_state.resample(inds)

            # ===== Update parameters ===== #
            to_accept, prop_state, prop_filt = run_pmmh(
                filter_,
                state.filter_state,
                dist,
                prop_filt,
                y,
                filter_._n_parallel
            )

            # ===== Update the description ===== #
            self.accepted = to_accept.sum().float() / float(to_accept.shape[0])

            filter_.exchange(prop_filt, to_accept)
            state.filter_state.exchange(prop_state, to_accept)
            state.w[:] = 0.0

        return self
