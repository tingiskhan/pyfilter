from typing import Callable, Tuple

import torch
from stochproc.timeseries import TimeseriesState

from .linear import LinearGaussianObservations

Fun = Callable[[TimeseriesState, Tuple[torch.Tensor, ...]], torch.Tensor]


# TODO: Fix this...
class LocalLinearization(LinearGaussianObservations):
    r"""
    Proposal utilizing a local linearization of the observation dynamics around the mean of the latent process. That is,
    given SSM with the following dynamics
        .. math::
            Y_t &= \alpha(X_t) + W_t, \newline
            X_{t+1} &= \gamma(X_t) + \delta(X_t) V_t,

    where both :math:`W_t, V_t` are two independent Gaussian distributions, we linearize the observation process s.t.
        .. math::
            \alpha(X_t) \approx \alpha(\mu_t) + \frac{d \alpha}{d x}(\mu_t) \left (X_t - \mu_t \right ).

    Collecting the terms allows us to approximate :math:`Y_t` as
        .. math::
            Y_t \approx b + A \dot X_t \dots,

    which in turn allows us to use :class:`LinearGaussianObservations`.
    """

    def __init__(self, f: Fun, linearized_f: Fun, s_index=-1, is_variance=False):
        r"""
        Initializes the :class:`LocalLinearization` class.

        Args:
            f: corresponds to :math:`\alpha`.
            linearized_f: corresponds to :math:`\frac{d \alpha}{d x}`.
        """

        raise NotImplementedError("Currently does not work!")

        super().__init__()
        self._f = f
        self._linearized_f = linearized_f

    def get_offset_and_scale(
        self, x: TimeseriesState, parameters: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, _ = self._model.hidden.mean_scale(x)

        mu_t = x.propagate_from(mean)
        d_alpha = self._linearized_f(mu_t, *parameters)

        if self._model.hidden.n_dim == 0:
            prod = d_alpha * mean
        else:
            # TODO: Need to verify this with test...
            prod = (d_alpha @ mean.unsqueeze(-1)).squeeze(-1)

        loc = self._f(mu_t, *parameters) - prod

        return d_alpha, loc

    def copy(self) -> "LocalLinearization":
        return LocalLinearization(self._f, self._linearized_f, self._s_index, is_variance=self._is_variance)
