import torch
from .base import BaseKernel
from .jittering import JitterKernel, NonShrinkingKernel
from ...utils import params_to_tensor, params_from_tensor


class OnlineKernel(BaseKernel):
    def __init__(self, kernel=None, discrete=False, **kwargs):
        """
        Base class for kernels being used in an online manner.
        :param kernel: The kernel density estimator to use
        :type kernel: JitterKernel
        """

        super().__init__(**kwargs)

        self._kernel = kernel or NonShrinkingKernel()
        self._disc = discrete

    def update(self, filter_, state, *args):
        weights = state.normalized_weights()

        stacked = params_to_tensor(filter_.ssm, constrained=False)
        indices = self._resampler(weights, normalized=True)

        jittered = self._kernel.jitter(stacked, weights, indices)

        filter_.resample(indices)
        state.filter_state.resample(indices, entire_history=False)

        if self._disc:
            to_jitter = (
                torch.empty(jittered.shape[0], device=jittered.device)
                .bernoulli_(1 / weights.shape[0] ** 0.5)
                .unsqueeze(-1)
            )

            jittered = (1 - to_jitter) * stacked[indices] + to_jitter * jittered

        params_from_tensor(filter_.ssm, jittered, constrained=False)
        state.w[:] = 0.0

        return self
