from .base import BaseKernel
from .kde import KernelDensityEstimate, NonShrinkingKernel
import torch


class OnlineKernel(BaseKernel):
    def __init__(self, kde=None, discrete=False, **kwargs):
        """
        Base class for kernels being used in an online manner.
        :param kde: The kernel density estimator to use
        :type kde: KernelDensityEstimate
        """

        super().__init__(**kwargs)

        self._kde = kde or NonShrinkingKernel()
        self._disc = discrete

    def _update(self, filter_, state, *args):
        weights = state.normalized_weights()

        stacked = filter_.ssm.parameters_to_array(constrained=False)
        kde = self._kde.fit(stacked, weights)

        inds = self._resampler(weights, normalized=True)
        filter_.resample(inds)
        state.filter_state.resample(inds, entire_history=False)

        jittered = kde.sample(inds=inds)

        if self._disc:
            to_jitter = (
                torch.empty(jittered.shape[0], device=jittered.device)
                .bernoulli_(1 / weights.shape[0] ** 0.5)
                .unsqueeze(-1)
            )

            jittered = (1 - to_jitter) * stacked[inds] + to_jitter * jittered

        filter_.ssm.parameters_from_array(jittered, constrained=False)
        state.w[:] = 0.0

        return self
