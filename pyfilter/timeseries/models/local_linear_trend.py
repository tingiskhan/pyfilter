import torch
from torch.distributions import Normal, AffineTransform, TransformedDistribution
from ..affine import AffineProcess
from ...typing import ArrayType
from ...utils import concater
from ...distributions import DistributionWrapper


def mean(x, a, sigma, _):
    return a.matmul(x.values.unsqueeze(-1)).squeeze(-1)


def scale(x, a, sigma, _):
    return sigma.pow(2).cumsum(-1).sqrt()


def initial_transform(module, base_dist):
    scale_, initial_mean = tuple(module.functional_parameters())[-2:]
    return TransformedDistribution(base_dist, AffineTransform(initial_mean, scale_.pow(2).cumsum(-1).sqrt()))


class LocalLinearTrend(AffineProcess):
    """
    Implements a Local Linear Trend model, i.e. a model with the following dynamics
        .. math::
            L_{t+1} = L_t + S_t + \\sigma_l W_{t+1}, \n
            S_{t+1} = S_t + \\sigma_s V_{t+1},

    where :math:`\\sigma_i > 0``, and :math:`W_t, V_t` are two independent zero mean and unit variance Gaussians. We
    also impose the restriction that :math:`\\sigma_l >= \\sigma_s`` by modelling :math:`\\sigma_l` as
        .. math::
            \\sigma_l^2 = \\sigma_s^2 + \\Delta \\sigma^2.
    """

    def __init__(self, sigma: ArrayType, initial_mean: ArrayType = torch.zeros(2), **kwargs):
        """
        Initializes the ``LocalLinearTrend`` class.

        Args:
            sigma: Corresponds to the vector :math:`[ \\sigma_s, \\Delta \\sigma ]``.
            initial_mean: Optional, specifies the initial mean.
            kwargs: See base.
        """

        parameters = (torch.tensor([[1.0, 0.0], [1.0, 1.0]]), sigma, initial_mean)
        initial_dist = increment_dist = DistributionWrapper(Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1)

        super().__init__(
            (mean, scale), parameters, initial_dist, increment_dist, initial_transform=initial_transform, **kwargs
        )


def semi_mean(x, mean_, coef_, sigma, _):
    slope = x.values[..., 0]

    new_level = x.values[..., 1] + slope
    new_slope = slope + coef_ * (mean_ - slope)

    return concater(new_slope, new_level)


def semi_scale(x, alpha, beta, sigma, _):
    return sigma.pow(2).cumsum(-1).sqrt()


def semi_initial_transform(module, base_dist):
    mean_, coef_, scale_, initial_mean = tuple(module.functional_parameters())

    var = scale_.pow(2)
    scale_ = concater(var[..., 0] / (2 * coef_), var[..., 1]).cumsum(-1).sqrt()

    return TransformedDistribution(base_dist, AffineTransform(initial_mean, scale_))


class SemiLocalLinearTrend(AffineProcess):
    """
    Implements a Semi Local Linear Trend model, which is a version of the Local Linear Trend model, but where we replace
    the random walk process of :math:`S` with a mean reverting one
        .. math::
            S_{t+1} = S_t + \\alpha (\\beta - S_t) + \\sigma_s V_{t+1}, \n
            S_0 \\sim \\mathcal{N}(x_0, \\frac{\\sigma_s}{\\sqrt{2\\beta}},

    where :math:`\\alpha \\in \\mathbb{R}``, and :math:`\\beta \\in (-1, 1)``.
    """

    def __init__(
        self, mean_: ArrayType, coef_: ArrayType, sigma: ArrayType, initial_mean: ArrayType = torch.zeros(2), **kwargs
    ):
        """
        Initializes the ``SemiLocalLinearTrend`` class.

        Args:
            mean_: Corresponds to :math:`\\alpha`.
            coef_: Corresponds to :math:`\\beta`.
            sigma: See ``LocalLinearTrend``.
            initial_mean: See ``LocalLinearTrend``.
            kwargs: See base.
        """

        parameters = (mean_, coef_, sigma, initial_mean)
        initial_dist = increment_dist = DistributionWrapper(Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1)

        super(SemiLocalLinearTrend, self).__init__(
            (semi_mean, semi_scale),
            parameters,
            initial_dist=initial_dist,
            increment_dist=increment_dist,
            initial_transform=semi_initial_transform,
            **kwargs
        )
