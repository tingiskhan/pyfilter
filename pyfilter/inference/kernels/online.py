from .base import BaseKernel
from ...kde import KernelDensityEstimate, ShrinkingKernel, NonShrinkingKernel, MultivariateGaussian
from ..utils import stacker
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
        self._th = ess

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

        if get_ess(weights, normalized=True) > self._th * weights.numel() and not (weights == 0.).any():
            return torch.arange(weights.numel())

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

    def _update(self, parameters, filter_, weights, log_weights):
        # ===== Define stacks ===== #
        stacked = stacker(parameters, lambda u: u.t_values)

        # ===== Check "convergence" ====== #
        w = add_dimensions(weights, stacked.concated.dim())

        mean = (w * stacked.concated).sum(0)
        var = (w * (stacked.concated - mean) ** 2).sum(0)

        if self._switched is None:
            self._switched = torch.zeros_like(mean).bool()

        if self._old_var is None:
            var_diff = var
        else:
            var_diff = var - self._old_var

        self._old_var = var
        self._switched = (var_diff.abs() < self._eps) & ~self._switched

        # ===== Resample ===== #
        inds = self._resample(filter_, weights, log_weights)

        # ===== Perform shrinkage ===== #
        jittered = torch.empty_like(stacked.concated)

        if (~self._switched).any():
            shrink_kde = self._shrink_kde.fit(stacked.concated[:, ~self._switched], weights)
            jittered[:, ~self._switched] = shrink_kde.sample(inds=inds)

        if self._switched.any():
            non_shrink = self._non_shrink.fit(stacked.concated[:, self._switched], weights)
            jittered[:, self._switched] = non_shrink.sample(inds=inds)

        # ===== Set new values ===== #
        for p, msk, ps in zip(parameters, stacked.mask, stacked.prev_shape):
            p.t_values = unflattify(jittered[:, msk], ps)

        return self


class KernelDensitySampler(BaseKernel):
    def __init__(self, kde=None, **kwargs):
        """
        Implements a kernel that samples from a KDE.
        :param kde: The KDE
        :type kde: KernelDensityEstimate
        """
        super().__init__(**kwargs)
        self._kde = kde or MultivariateGaussian()

    def _update(self, parameters, filter_, weights, log_weights):
        stacked = stacker(parameters, lambda u: u.t_values)

        # ===== Calculate covariance ===== #
        kde = self._kde.fit(stacked.concated, weights)

        # ===== Resample ===== #
        inds = self._resampler(weights, normalized=True)
        filter_.resample(inds, entire_history=False)

        # ===== Sample params ===== #
        samples = kde.sample(inds=inds)
        for p, msk, ps in zip(parameters, stacked.mask, stacked.prev_shape):
            p.t_values = unflattify(samples[:, msk], ps)

        log_weights[:] = 0.

        return self