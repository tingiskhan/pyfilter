import torch
from torch.distributions import Independent, Normal, TransformedDistribution, AffineTransform, Distribution
from ...typing import ArrayType
from ...distributions import DistributionWrapper
from ..affine import AffineProcess
from ...utils import concater


def f(x, sigma_volatility, _):
    return x.values


def g(x, sigma_volatility, _):
    x1 = x.values[..., 1].exp()
    x2 = sigma_volatility

    return concater(x1, x2)


# TODO: Figure out initial density?
def initial_transform(model: "UCSV", base_dist: Distribution):
    sigma_volatility, initial_state_mean = model.functional_parameters()

    transform = AffineTransform(
        initial_state_mean if initial_state_mean is not None else base_dist.mean,
        concater(sigma_volatility, sigma_volatility),
    )

    return TransformedDistribution(base_dist, transform, validate_args=False)


class UCSV(AffineProcess):
    """
    Implements a UCSV model, i.e. a stochastic process with the dynamics
        .. math::
            L_{t+1} = L_t + V_t W_{t+1}, \n
            \\log{V_{t+1}} = \\log{V_t} + \\sigma_v U_{t+1}, \n
            L_0, \\log{V_0} \\sim \\mathcal{N}(x^i_0, \\sigma_v), \\: i \\in [L, V].

    where :math:`\\sigma_v > 0`.
    """

    def __init__(self, sigma_volatility, initial_state_mean: ArrayType = None, **kwargs):
        """
        Inititalizes the ``UCSV`` class.

        Args:
            sigma_volatility: The volatility of the log volatility process, i.e. :math:`\\sigma_v`.
            initial_state_mean: Optional, whether to use initial values other than 0 for both processes.
            kwargs: See base.
        """

        initial_dist = increment_dist = DistributionWrapper(
            lambda **u: Independent(Normal(**u), 1), loc=torch.zeros(2), scale=torch.ones(2)
        )

        super().__init__(
            (f, g),
            (sigma_volatility, initial_state_mean),
            initial_dist,
            increment_dist,
            initial_transform=initial_transform,
            **kwargs
        )
