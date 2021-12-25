import torch
from torch.distributions import Normal
from ...typing import ArrayType
from ...distributions import DistributionWrapper
from ..affine import AffineProcess
from ...utils import concater


def f(x, sigma_volatility):
    return x.values


def g(x, sigma_volatility):
    x1 = x.values[..., 1].exp()
    x2 = sigma_volatility

    return concater(x1, x2)


class UCSV(AffineProcess):
    """
    Implements a UCSV model, i.e. a stochastic process with the dynamics
        .. math::
            L_{t+1} = L_t + V_t W_{t+1}, \n
            \\log{V_{t+1}} = \\log{V_t} + \\sigma_v U_{t+1}, \n
            L_0, \\log{V_0} \\sim \\mathcal{N}(x^i_0, \\sigma^i_0), \\: i \\in [L, V].

    where :math:`\\sigma_v > 0`.
    """

    def __init__(
        self,
        sigma_volatility,
        initial_state_mean: ArrayType = torch.zeros(2),
        initial_state_scale: ArrayType = torch.ones(2),
        **kwargs
    ):
        """
        Inititalizes the ``UCSV`` class.

        Args:
            sigma_volatility: The volatility of the log volatility process, i.e. :math:`\\sigma_v`.
            initial_state_mean: Optional, whether to use initial values other than 0 for both processes.
            kwargs: See base.
        """

        increment_dist = DistributionWrapper(
            Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1
        )

        initial_dist = DistributionWrapper(
            Normal, loc=initial_state_mean, scale=initial_state_scale, reinterpreted_batch_ndims=1
        )

        super().__init__((f, g), (sigma_volatility,), initial_dist, increment_dist, **kwargs)
