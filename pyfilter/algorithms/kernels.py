from ..filters.base import BaseFilter
from ..utils import get_ess, normalize
from ..timeseries.parameter import Parameter
import torch
from scipy.stats import chi2
from math import sqrt


class BaseKernel(object):
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

        raise NotImplementedError()


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
    if values.dim() > w.dim():
        w = w.unsqueeze(-1)

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
    def update(self, parameters, filter_, weights):
        normalized = normalize(weights)
        ess = get_ess(normalized, normalized=True)

        # ===== Perform shrinkage ===== #
        ms_hid, ms_obs = filter_.ssm.p_map(lambda u: _shrink(u.t_values, normalized, ess))
        meanscales = ms_hid + ms_obs

        # ===== Mutate parameters ===== #
        for p, (m, s) in zip(parameters, meanscales):
            p.t_values = _jitter(m, s)

        return self


class AdaptiveShrinkageKernel(BaseKernel):
    def __init__(self, p=4, vthresh_scale=1.):
        """
        Implements the adaptive shrinkage kernel of ..
        :param p: The parameter p controlling the jittering variance.
        :type p: float
        :param vthresh_scale: The scaling of the threshold of the variance
        :type vthresh_scale: float
        """

        self._vn = vthresh_scale
        self._vf = self._vn / 4
        self._p = p
        self._switched = False

    def update(self, parameters, filter_, weights):
        normalized = normalize(weights)
        ess = get_ess(normalized, normalized=True)

        # ===== Perform shrinkage ===== #
        ms_hid, ms_obs = filter_.ssm.p_map(lambda u: _shrink(u.t_values, normalized, ess))
        meanscales = ms_hid + ms_obs

        # ===== Check if to switch ===== #
        # TODO: This should be moved outside
        scale = sqrt(weights.numel() ** (-(self._p + 2) / self._p))
        if not self._switched:
            self._switched = min(s for m, s in meanscales) < self._vn * scale

        # ===== Mutate parameters ===== #
        for p, (m, s) in zip(parameters, meanscales):
            p.t_values = _jitter(
                m if not self._switched else p.t_values,
                s if not self._switched else min(max(s, self._vf * scale), self._vn * scale)
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