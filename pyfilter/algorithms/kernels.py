from ..filters.base import BaseFilter
from ..utils import get_ess, normalize, add_dimensions
from ..timeseries.parameter import Parameter
import torch
from scipy.stats import chi2
from math import sqrt
from torch.distributions import MultivariateNormal, Independent


class BaseKernel(object):
    def __init__(self, record_stats=True, length=2):
        """
        The base kernel used for propagating parameters.
        :param record_stats: Whether to record the statistics
        :type record_stats: bool
        :param length: How many to record. If `None` records everything
        :type length: int
        """
        self._record_stats = record_stats

        self._recorded_stats = tuple()
        self._length = length

    def _update(self, parameters, filter_, weights):
        """
        Defines the function for updating the parameters for the user to override.
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

        self._update(parameters, filter_, w)

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

        res = tuple()
        for p in parameters:
            weights = add_dimensions(weights, p.dim())
            vals = p.t_values

            mean = (vals * weights).sum(0)
            scale = ((vals - mean) ** 2 * weights).sum(0).sqrt()

            res += ({
                'mean': mean,
                'scale': scale
            },)

        self._recorded_stats += (
            res,
        )

        if self._length is not None:
            self._recorded_stats = self._recorded_stats[-self._length:]

        return self

    def get_as_numpy(self):
        """
        Returns the stats numpy arrays instead of torch tensor.
        :rtype: tuple[tuple[dict[str, numpy.array]]
        """
        res = tuple()
        for pt in self._recorded_stats:
            temp = tuple()
            for p in pt:
                tempdict = dict()
                for k, v in p.items():
                    tempdict[k] = v.numpy()

                temp += (tempdict,)
            res += (temp,)

        return res

    def get_diff(self):
        """
        Get the difference between the two latest scales.
        :rtype: tuple[torch.Tensor]
        """

        if self._length is not None and self._length < 2:
            raise ValueError('Length must be bigger than 1!')
        elif len(self._recorded_stats) < 2:
            return tuple(p['scale'] for p in self._recorded_stats[-1])

        res = tuple()
        for pt, ptm1 in zip(self._recorded_stats[-1], self._recorded_stats[-2]):
            res += (pt['scale'] - ptm1['scale'],)

        return res


def _normal_test(x, alpha=0.05):
    """
    Implements a basic Jarque-Bera test for normality.
    :param x: The data
    :type x: torch.Tensor
    :param alpha: The level of confidence
    :type alpha: float
    :return: Whether a normal distribution or not
    :rtype: bool
    """
    mean = x.mean()
    var = ((x - mean) ** 2).mean()

    # ===== Skew ===== #
    skew = ((x - mean) ** 3).mean() / var ** 1.5

    # ===== Kurtosis ===== #
    kurt = ((x - mean) ** 4).mean() / var ** 2

    # ===== Statistic ===== #
    jb = x.shape[0] / 6 * (skew ** 2 + 1 / 4 * (kurt - 3) ** 2)

    if chi2(2).ppf(1 - alpha) < jb:
        return False

    return True


def _jitter(values, scale):
    """
    Jitters the parameters.
    :param values: The values
    :type values: torch.Tensor
    :param scale: The scaling to use for the variance of the proposal
    :type scale: float
    :return: Proposed values
    :rtype: torch.Tensor
    """

    return values + scale * torch.empty_like(values).normal_()


def _continuous_jitter(parameter, w, p, ess, shrink=True):
    """
    Jitters the parameters using the optimal shrinkage of ...
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param w: The normalized weights
    :type w: torch.Tensor
    :param p: The p parameter
    :type p: float
    :param ess: The previous ESS
    :type ess: float
    :param shrink: Whether to shrink as well as adjusting variance
    :type shrink: bool
    :return: Proposed values
    :rtype: torch.Tensor
    """
    values = parameter.t_values

    if not shrink:
        return _jitter(values, 1 / sqrt(ess ** ((p + 2) / p)))

    mean, bw = _shrink(values, w, ess)

    return _jitter(mean, bw)


def _shrink(values, w, ess):
    """
    Shrinks the parameters towards their mean.
    :param values: The values
    :type values: torch.Tensor
    :param w: The normalized weights
    :type w: torch.Tensor
    :param p: The p parameter
    :type p: float
    :param ess: The previous ESS
    :type ess: float
    :return: The mean of the shrunk distribution and bandwidth
    :rtype: torch.Tensor, torch.Tensor
    """
    # ===== Calculate mean ===== #
    w = add_dimensions(w, values.dim())

    mean = (w * values).sum(0)

    # ===== Calculate STD ===== #
    if not _normal_test(values):
        sort, _ = values.sort(0)
        std = (sort[int(0.75 * values.shape[0])] - sort[int(0.25 * values.shape[0])]) / 1.349

        var = std ** 2
    else:
        var = (w * (values - mean) ** 2).sum(0)
        std = var.sqrt()

    # ===== Calculate bandwidth ===== #
    bw = 1.59 * std * ess ** (-1 / 3)

    # ===== Calculate shrinkage and shrink ===== #
    beta = ((var - bw ** 2) / var).sqrt()

    return mean + beta * (values - mean), bw


class ShrinkageKernel(BaseKernel):
    """
    An improved regular shrinkage kernel, from the paper ..
    """

    def _update(self, parameters, filter_, weights):
        ess = get_ess(weights, normalized=True)

        # ===== Perform shrinkage ===== #
        ms_hid, ms_obs = filter_.ssm.p_map(lambda u: _shrink(u.t_values, weights, ess))
        meanscales = ms_hid + ms_obs

        # ===== Mutate parameters ===== #
        for p, (m, s) in zip(parameters, meanscales):
            p.t_values = _jitter(m, s)

        return self


class AdaptiveShrinkageKernel(BaseKernel):
    def __init__(self, eps=1e-4, **kwargs):
        """
        Implements the adaptive shrinkage kernel of ..
        :param eps: The tolerance for when to stop shrinking
        :type eps: float
        """

        super().__init__(**kwargs)
        self._eps = eps
        self._switched = False

    def _update(self, parameters, filter_, weights):
        ess = get_ess(weights, normalized=True)

        # ===== Perform shrinkage ===== #
        ms_hid, ms_obs = filter_.ssm.p_map(lambda u: _shrink(u.t_values, weights, ess))
        meanscales = ms_hid + ms_obs

        # ===== Mutate parameters ===== #
        for p, (m, s), delta in zip(parameters, meanscales, self.get_diff()):
            switched = (delta.abs() < self._eps).all()

            p.t_values = _jitter(
                m if not switched else p.t_values,
                s
            )

        return self


def _disc_jitter(parameter, i, w, p, ess, shrink):
    """
    Jitters the parameters using discrete propagation.
    :param parameter: The parameters of the model, inputs as (values, prior)
    :type parameter: Parameter
    :param i: The indices to jitter
    :type i: torch.Tensor
    :param w: The normalized weights
    :type w: torch.Tensor
    :param p: The p parameter
    :type p: float
    :param ess: The previous ESS
    :type ess: float
    :param shrink: Whether to shrink as well as adjusting variance
    :type shrink: bool
    :return: Proposed values
    :rtype: torch.Tensor
    """
    # TODO: This may be improved
    if i.sum() == 0:
        return parameter.t_values

    return (1 - i) * parameter.t_values + i * _continuous_jitter(parameter, w, p, ess, shrink=shrink)


def _mcmc_move(params, dist):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params: The parameters to use for defining the distribution
    :type params: tuple[Distribution]
    :param dist: The distribution to use for sampling
    :type dist: MultivariateNormal
    :return: Samples from a multivariate normal distribution
    :rtype: torch.Tensor
    """
    shape = next(p.t_values.shape for p in params)
    if len(shape) > 1:
        shape = shape[:-1]

    rvs = dist.sample(shape)

    for p, vals in zip(params, rvs.t()):
        p.t_values = vals

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

    p_vals = torch.stack([p.t_values for p in params], dim=-1)
    n_p_vals = torch.stack([p.t_values for p in n_params], dim=-1)

    return dist.log_prob(p_vals) - dist.log_prob(n_p_vals)


class ParticleMetropolisHastings(BaseKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._resampler = None
        self._y = None
        self.accepted = None

    def set_resampler(self, resampler):
        self._resampler = resampler

    def set_data(self, y):
        self._y = y

    def define_pdf(self, parameters, weights):
        """
        The method to be overriden by the user for defining the kernel to propagate the parameters.
        :return: A distribution
        :rtype: MultivariateNormal|Independent
        """

        raise NotImplementedError()

    def _update(self, parameters, filter_, weights):
        # ===== Construct distribution ===== #
        dist = self.define_pdf(parameters, weights)

        # ===== Resample among parameters ===== #
        inds = self._resampler(weights)
        filter_.resample(inds, entire_history=True)

        # ===== Define new filters and move via MCMC ===== #
        t_filt = filter_.copy()
        _mcmc_move(t_filt.ssm.flat_theta_dists, dist)

        # ===== Filter data ===== #
        t_filt.reset().initialize().to_(filter_._device).longfilter(self._y, bar=False)

        quotient = t_filt.loglikelihood - filter_.loglikelihood

        # ===== Calculate acceptance ratio ===== #
        plogquot = t_filt.ssm.p_prior() - filter_.ssm.p_prior()
        kernel = _eval_kernel(filter_.ssm.flat_theta_dists, dist, t_filt.ssm.flat_theta_dists)

        # ===== Check which to accept ===== #
        u = torch.empty_like(quotient).uniform_().log()
        toaccept = u < quotient + plogquot + kernel

        # ===== Update the description ===== #
        self.accepted = toaccept.sum().float() / float(toaccept.shape[0])
        filter_.exchange(t_filt, toaccept)

        return self


class SymmetricMH(ParticleMetropolisHastings):
    def define_pdf(self, parameters, weights):
        asarray = torch.stack([p.t_values for p in parameters], dim=-1)

        if asarray.dim() > 2:
            asarray = asarray[..., 0]

        mean = (asarray * weights.unsqueeze(-1)).sum(0)
        centralized = asarray - mean
        cov = torch.matmul(weights * centralized.t(), centralized)

        return MultivariateNormal(mean, scale_tril=torch.cholesky(cov))
