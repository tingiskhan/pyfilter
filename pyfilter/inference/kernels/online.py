from .base import BaseKernel
from ...kde import KernelDensityEstimate, NonShrinkingKernel
from ..utils import stacker
from ...utils import unflattify
import torch


class OnlineKernel(BaseKernel):
    def __init__(self, kde=None, **kwargs):
        """
        Base class for kernels being used in an online manner where updates are performed at each time step.
        :param kde: The KDE algorithm to use
        :type kde: KernelDensityEstimate
        """
        super().__init__(**kwargs)

        self._kde = kde or NonShrinkingKernel()

    def _resample(self, filter_, weights, log_weights):
        """
        Helper method for performing resampling.
        :param filter_: The filter to resample
        :type filter_: BaseFilter
        :param weights: The weights
        :type weights: torch.Tensor
        :param log_weights: The log-weights to update if resampling
        :type log_weights: torch.Tensor
        :rtype: torch.Tensor
        """

        inds = self._resampler(weights, normalized=True)
        filter_.resample(inds, entire_history=False)
        log_weights[:] = 0.

        return inds

    def _update(self, parameters, filter_, weights, log_weights):
        # ===== Perform shrinkage ===== #
        stacked = stacker(parameters, lambda u: u.t_values)
        kde = self._kde.fit(stacked.concated, weights)

        inds = self._resample(filter_, weights, log_weights)
        jittered = kde.sample(inds=inds)

        # ===== Mutate parameters ===== #
        for p, msk, ps in zip(parameters, stacked.mask, stacked.prev_shape):
            p.t_values = unflattify(jittered[:, msk], ps)

        return self
