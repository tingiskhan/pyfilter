from .base import BaseKernel
from ...kde import KernelDensityEstimate, NonShrinkingKernel
from ..utils import stacker
from ...utils import unflattify
import torch
from ...filters.base import BaseFilter


class OnlineKernel(BaseKernel):
    def __init__(self, kde=None, **kwargs):
        """
        Base class for kernels being used in an online manner.
        :param kde: The kernel density estimator to use
        :type kde: KernelDensityEstimate
        """
        super().__init__(**kwargs)

        self._kde = kde or NonShrinkingKernel()

    def _resample(self, filter_: BaseFilter, weights: torch.Tensor):
        """
        Helper method for performing resampling.
        :param filter_: The filter to resample
        :param weights: The weights
        """

        inds = self._resampler(weights, normalized=True)
        filter_.resample(inds, entire_history=False)

        return inds

    def _update(self, parameters, filter_, weights):
        # ===== Perform shrinkage ===== #
        stacked = stacker(parameters, lambda u: u.t_values)
        kde = self._kde.fit(stacked.concated, weights)

        inds = self._resample(filter_, weights)
        jittered = kde.sample(inds=inds)

        # ===== Mutate parameters ===== #
        for p, msk, ps in zip(parameters, stacked.mask, stacked.prev_shape):
            p.t_values = unflattify(jittered[:, msk], ps)

        return self
