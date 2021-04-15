from .base import BaseKernel
from ...utils import _construct_mvn, PropConstructor, run_pmmh, params_to_tensor


class SymmetricMH(object):
    def __call__(self, state, filter_, y):
        values = params_to_tensor(filter_.ssm, constrained=False)
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
        self._proposal = proposal or SymmetricMH()
        self.accepted = None

    def _update(self, filter_, state, y, *args):
        prop_filt = filter_.copy()

        for _ in range(self._n_steps):
            inds = self._resampler(state.normalized_weights(), normalized=True)
            dist = self._proposal(state, filter_, y)

            filter_.resample(inds)
            state.filter_state.resample(inds)

            to_accept, prop_state, prop_filt = run_pmmh(
                filter_, state.filter_state, dist, prop_filt, y, filter_.n_parallel
            )

            self.accepted = to_accept.sum().float() / float(to_accept.shape[0])

            filter_.exchange(prop_filt, to_accept)
            state.filter_state.exchange(prop_state, to_accept)
            state.w[:] = 0.0

        return self
