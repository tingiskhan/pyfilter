import torch

from .base import BaseKernel
from .jittering import JitterKernel, NonShrinkingKernel


class OnlineKernel(BaseKernel):
    """
    Kernel for mutating parameter particles in an online fashion.
    """

    def __init__(self, kernel: JitterKernel = None, discrete=False, **kwargs):
        """
        Internal initializer for :class:`OnlineKernel`.

        Args:
            kernel (JitterKernel, optional): kernel to use for jittering the parameter particles.. Defaults to None.
            discrete (bool, optional): whether to mutate all particles, or just some of them with a probability proportional to ESS.. Defaults to False.
        """

        super().__init__(**kwargs)

        self._kernel = kernel or NonShrinkingKernel()
        self._disc = discrete

    def update(self, context, filter_, state):
        weights = state.normalized_weights()

        stacked = context.stack_parameters(constrained=False)
        indices = self._resampler(weights, normalized=True)

        jittered = self._kernel.jitter(stacked, weights, indices)

        context.resample(indices)
        state.filter_state.resample(indices, entire_history=False)

        if self._disc:
            to_jitter = (
                torch.empty(jittered.shape[0], device=jittered.device)
                .bernoulli_(1 / weights.shape[0] ** 0.5)
                .unsqueeze(-1)
            )

            jittered = (1 - to_jitter) * stacked[indices] + to_jitter * jittered

        context.unstack_parameters(jittered, constrained=False)
        
        with context.no_prior_verification():
            filter_.initialize_model(context)

        state.w.fill_(0.0)

        return state
