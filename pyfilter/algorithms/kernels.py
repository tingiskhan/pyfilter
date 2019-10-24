from ..filters.base import BaseFilter
from ..utils import normalize, add_dimensions, unflattify, get_ess
from ..timeseries.parameter import Parameter
import torch
from torch.distributions import MultivariateNormal, Independent
import numpy as np
from ..kde import KernelDensityEstimate, Gaussian, ShrinkingKernel, NonShrinkingKernel
from ..resampling import residual


def stacker(parameters, selector=lambda u: u.values):
    """
    Stacks the parameters and returns a n-tuple containing the mask for each parameter.
    :param parameters: The parameters
    :type parameters: tuple[Parameter]|list[Parameter]
    :param selector: The selector
    :rtype: torch.Tensor, tuple[slice]
    """

    to_conc = tuple()
    mask = tuple()

    i = 0
    for p in parameters:
        if p.c_numel() < 2:
            to_conc += (selector(p).unsqueeze(-1),)
            slc = i
        else:
            to_conc += (selector(p).flatten(1),)
            slc = slice(i, i + p.c_numel())

        mask += (slc,)
        i += p.c_numel()

    return torch.cat(to_conc, dim=-1), mask


def _construct_mvn(x, w):
    """
    Constructs a multivariate normal distribution of weighted samples.
    :param x: The samples
    :type x: torch.Tensor
    :param w: The weights
    :type w: torch.Tensor
    :rtype: MultivariateNormal
    """

    mean = (x * w.unsqueeze(-1)).sum(0)
    centralized = x - mean
    cov = torch.matmul(w * centralized.t(), centralized)

    return MultivariateNormal(mean, scale_tril=torch.cholesky(cov))


class BaseKernel(object):
    def __init__(self, record_stats=True, resampling=residual, ess=0.9):
        """
        The base kernel used for propagating parameters.
        :param record_stats: Whether to record the statistics
        :type record_stats: bool
        :param ess: At which ESS to resample
        :type ess: ess
        """

        self._record_stats = record_stats

        self._recorded_stats = dict()
        self._recorded_stats['mean'] = tuple()
        self._recorded_stats['scale'] = tuple()

        self._resampler = resampling
        self._th = ess

    def set_resampler(self, resampler):
        """
        Sets the resampler to use if necessary for kernel.
        :param resampler: The resampler
        :type resampler: callable
        :rtype: BaseKernel
        """
        self._resampler = resampler

        return self

    def _update(self, parameters, filter_, weights):
        """
        Defines the function for updating the parameters for the user to override. Should return whether it resampled or
        not.
        :param parameters: The parameters of the model to update
        :type parameters: tuple[Parameter]
        :param filter_: The filter
        :type filter_: BaseFilter
        :param weights: The weights to be passed
        :type weights: torch.Tensor
        :return: Self
        :rtype: BaseKernel
        """

        raise NotImplementedError()

    def update(self, parameters, filter_, weights):
        """
        Defines the function for updating the parameters.
        :param parameters: The parameters of the model to update
        :type parameters: tuple[Parameter]
        :param filter_: The filter
        :type filter_: BaseFilter
        :param weights: The weights to be passed
        :type weights: torch.Tensor
        :return: Self
        :rtype: BaseKernel
        """

        w = normalize(weights)

        if self._record_stats:
            self.record_stats(parameters, w)

        resampled = self._update(parameters, filter_, w)
        if resampled:
            weights[:] = 0.

        return self

    def record_stats(self, parameters, weights):
        """
        Records the stats of the parameters.
        :param parameters: The parameters of the model to update
        :type parameters: tuple[Parameter]
        :param weights: The weights to be passed
        :type weights: torch.Tensor
        :return: Self
        :rtype: BaseKernel
        """

        m_res = tuple()
        s_res = tuple()
        for p in parameters:
            weights = add_dimensions(weights, p.dim())
            vals = p.t_values

            mean = (vals * weights).sum(0)
            scale = ((vals - mean) ** 2 * weights).sum(0).sqrt()

            m_res += (mean,)
            s_res += (scale,)

        self._recorded_stats['mean'] += (m_res,)
        self._recorded_stats['scale'] += (s_res,)

        return self

    def get_as_numpy(self):
        """
        Returns the stats numpy arrays instead of torch tensor.
        :rtype: dict[str,np.ndarray]
        """

        res = dict()
        for k, v in self._recorded_stats.items():
            t_res = tuple()
            for pt in v:
                t_res += (np.array(pt),)

            res[k] = np.stack(t_res)

        return res


class OnlineKernel(BaseKernel):
    def __init__(self, kde=None, **kwargs):
        """
        An improved regular shrinkage kernel, from the paper ..
        :param kde: The KDE algorithm to use
        :type kde: KernelDensityEstimate
        """
        super().__init__(**kwargs)

        self._kde = kde or ShrinkingKernel()
        self._resampled = None

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
    def __init__(self, eps=1e-4, **kwargs):
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
            shrink_kde._means = shrink_kde._means[inds]

            jittered[:, ~self._switched] = shrink_kde.sample()

        if self._switched.any():
            non_shrink = self._non_shrink.fit(stacked[:, self._switched], weights)
            non_shrink._means = non_shrink._means[inds]

            jittered[:, self._switched] = non_shrink.sample()

        # ===== Set new values ===== #
        for p, msk in zip(parameters, mask):
            p.t_values = unflattify(jittered[:, msk], p.c_shape)

        return self._resampled


class KernelDensitySampler(BaseKernel):
    def __init__(self, kde=None, **kwargs):
        """
        Implements a sampler that samples from a KDE representation.
        :param kde: The KDE
        :type kde: KernelDensityEstimate
        """
        super().__init__(**kwargs)
        self._kde = kde or Gaussian()

    def _update(self, parameters, filter_, weights):
        values, mask = stacker(parameters, lambda u: u.t_values)

        # ===== Calculate covariance ===== #
        kde = self._kde.fit(values, weights)

        # ===== Resample ===== #
        inds = self._resampler(weights, normalized=True)
        filter_.resample(inds, entire_history=False)

        # ===== Sample params ===== #
        samples = kde.sample()
        for p, msk in zip(parameters, mask):
            p.t_values = unflattify(samples[:, msk], p.c_shape)

        return True


def _mcmc_move(params, dist, mask, shape):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params: The parameters to use for defining the distribution
    :type params: tuple[Parameter]
    :param dist: The distribution to use for sampling
    :type dist: MultivariateNormal
    :param mask: The mask to apply for parameters
    :type mask: tuple[slice]
    :param shape: The shape to sample
    :type shape: int
    :return: Samples from a multivariate normal distribution
    :rtype: torch.Tensor
    """

    rvs = dist.sample((shape,))

    for p, msk in zip(params, mask):
        p.t_values = unflattify(rvs[:, msk], p.c_shape)

    return True


def _eval_kernel(params, dist, n_params):
    """
    Evaluates the kernel used for performing the MCMC move.
    :param params: The current parameters
    :type params: tuple[Distribution]
    :param dist: The distribution to use for evaluating the prior
    :type dist: MultivariateNormal
    :param n_params: The new parameters to evaluate against
    :type n_params: tuple of Distribution
    :return: The log difference in priors
    :rtype: torch.Tensor
    """

    p_vals, _ = stacker(params, lambda u: u.t_values)
    n_p_vals, _ = stacker(n_params, lambda u: u.t_values)

    return dist.log_prob(p_vals) - dist.log_prob(n_p_vals)


class ParticleMetropolisHastings(BaseKernel):
    def __init__(self, nsteps=1, **kwargs):
        """
        Implements a base class for the particle Metropolis Hastings class.
        :param nsteps: The number of steps to perform
        :type nsteps: int
        """
        super().__init__(**kwargs)

        self._nsteps = nsteps
        self._y = None
        self.accepted = None

    def set_data(self, y):
        """
        Sets the data to be used when calculating acceptance probabilities.
        :param y: The data
        :type y: tuple[torch.Tensor]
        :return: Self
        :rtype: ParticleMetropolisHastings
        """
        self._y = y

        return self

    def define_pdf(self, values, weights):
        """
        The method to be overridden by the user for defining the kernel to propagate the parameters. Note that the
        parameters are propagated in the transformed space.
        :param values: The parameters as a single Tensor
        :type values: torch.Tensor
        :param weights: The normalized weights of the particles
        :type weights: torch.Tensor
        :return: A distribution
        :rtype: MultivariateNormal|Independent
        """

        raise NotImplementedError()

    def _update(self, parameters, filter_, weights):
        for i in range(self._nsteps):
            # ===== Construct distribution ===== #
            stacked, mask = stacker(parameters, lambda u: u.t_values)
            dist = self.define_pdf(stacked, weights)

            # ===== Resample among parameters ===== #
            inds = self._resampler(weights, normalized=True)
            filter_.resample(inds, entire_history=True)

            # ===== Define new filters and move via MCMC ===== #
            t_filt = filter_.copy()
            t_filt.viewify_params((filter_._n_parallel, 1))
            _mcmc_move(t_filt.ssm.theta_dists, dist, mask, stacked.shape[0])

            # ===== Filter data ===== #
            t_filt.reset().initialize().longfilter(self._y, bar=False)

            quotient = t_filt.loglikelihood - filter_.loglikelihood

            # ===== Calculate acceptance ratio ===== #
            plogquot = t_filt.ssm.p_prior() - filter_.ssm.p_prior()
            kernel = _eval_kernel(filter_.ssm.theta_dists, dist, t_filt.ssm.theta_dists)

            # ===== Check which to accept ===== #
            u = torch.empty_like(quotient).uniform_().log()
            toaccept = u < quotient + plogquot + kernel

            # ===== Update the description ===== #
            self.accepted = toaccept.sum().float() / float(toaccept.shape[0])
            filter_.exchange(t_filt, toaccept)

            weights = normalize(filter_.loglikelihood)

        return True


class SymmetricMH(ParticleMetropolisHastings):
    def define_pdf(self, values, weights):
        return _construct_mvn(values, weights)
