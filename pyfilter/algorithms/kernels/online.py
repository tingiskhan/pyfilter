from .base import BaseKernel
from ...kde import KernelDensityEstimate, ShrinkingKernel, NonShrinkingKernel, MultivariateGaussian
from pyfilter.algorithms.utils import stacker
from ...utils import unflattify, get_ess, add_dimensions
import torch


class OnlineKernel(BaseKernel):
    def __init__(self, kde=None, ess=0.9, **kwargs):
        """
        Base class for kernels being used in an online manner where updates are performed at each time step.
        :param kde: The KDE algorithm to use
        :type kde: KernelDensityEstimate
        :param ess: At which ESS to resample
        :type ess: ess
        """
        super().__init__(**kwargs)

        self._kde = kde or ShrinkingKernel()
        self._resampled = None
        self._th = ess

    def _resample(self, filter_, weights):
        """
        Helper method for performing resampling.
        :param filter_: The filter to resample
        :type filter_: BaseFilter
        :param weights: The weights
        :type weights: torch.Tensor
        :rtype: torch.Tensor
        """

        self._resampled = False

        if get_ess(weights, normalized=True) > self._th * weights.numel():
            return torch.arange(weights.numel())

        inds = self._resampler(weights, normalized=True)
        filter_.resample(inds, entire_history=False)
        self._resampled = True

        return inds

    def _update(self, parameters, filter_, weights):
        # ===== Perform shrinkage ===== #
        stacked, mask = stacker(parameters, lambda u: u.t_values)
        kde = self._kde.fit(stacked, weights)

        inds = self._resample(filter_, weights)
        kde._means = kde._means[inds]

        jittered = kde.sample()

        # ===== Mutate parameters ===== #
        for msk, p in zip(mask, parameters):
            p.t_values = unflattify(jittered[:, msk], p.c_shape)

        return self._resampled


# TODO: The eps is completely arbitrary... but kinda influences the posterior
class AdaptiveKernel(OnlineKernel):
    def __init__(self, eps=5e-5, **kwargs):
        """
        Implements the adaptive shrinkage kernel of ..
        :param eps: The tolerance for when to stop shrinking
        :type eps: float
        """
        super().__init__(**kwargs)
        self._eps = eps
        self._old_var = None
        self._switched = None

        self._shrink_kde = ShrinkingKernel()
        self._non_shrink = NonShrinkingKernel()

    def _update(self, parameters, filter_, weights):
        # ===== Define stacks ===== #
        stacked, mask = stacker(parameters, lambda u: u.t_values)

        # ===== Check "convergence" ====== #
        w = add_dimensions(weights, stacked.dim())

        mean = (w * stacked).sum(0)
        var = (w * (stacked - mean) ** 2).sum(0)

        if self._switched is None:
            self._switched = torch.zeros_like(mean).bool()

        if self._old_var is None:
            var_diff = var
        else:
            var_diff = var - self._old_var

        self._old_var = var
        self._switched = (var_diff.abs() < self._eps) & ~self._switched

        # ===== Resample ===== #
        inds = self._resample(filter_, weights)

        # ===== Perform shrinkage ===== #
        jittered = torch.empty_like(stacked)

        if (~self._switched).any():
            shrink_kde = self._shrink_kde.fit(stacked[:, ~self._switched], weights)
            jittered[:, ~self._switched] = shrink_kde.sample(inds=inds)

        if self._switched.any():
            non_shrink = self._non_shrink.fit(stacked[:, self._switched], weights)
            jittered[:, self._switched] = non_shrink.sample(inds=inds)

        # ===== Set new values ===== #
        for p, msk in zip(parameters, mask):
            p.t_values = unflattify(jittered[:, msk], p.c_shape)

        return self._resampled


class KernelDensitySampler(BaseKernel):
    def __init__(self, kde=None, **kwargs):
        """
        Implements a kernel that samples from a KDE.
        :param kde: The KDE
        :type kde: KernelDensityEstimate
        """
        super().__init__(**kwargs)
        self._kde = kde or MultivariateGaussian()

    def _update(self, parameters, filter_, weights):
        values, mask = stacker(parameters, lambda u: u.t_values)

        # ===== Calculate covariance ===== #
        kde = self._kde.fit(values, weights)

        # ===== Resample ===== #
        inds = self._resampler(weights, normalized=True)
        filter_.resample(inds, entire_history=False)

        # ===== Sample params ===== #
        samples = kde.sample(inds=inds)
        for p, msk in zip(parameters, mask):
            p.t_values = unflattify(samples[:, msk], p.c_shape)

        return True