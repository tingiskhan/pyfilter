from .base import BaseFilter
from abc import ABC
from ..resampling import systematic
from ..timeseries import LinearGaussianObservations as LGO
from ..proposals.bootstrap import Bootstrap, Proposal
import torch
from ..utils import get_ess, normalize
from ..module import TensorContainer
from .utils import _construct_empty
from typing import Tuple, Union
from ..proposals import LinearGaussianObservations


_PROPOSAL_MAPPING = {
    LGO: LinearGaussianObservations
}


class ParticleFilter(BaseFilter, ABC):
    def __init__(self, model, particles: int, resampling=systematic, proposal: Union[str, Proposal] = 'auto', ess=0.9,
                 need_grad=False, log_ess=True, **kwargs):
        """
        Implements the base functionality of a particle filter.
        :param particles: How many particles to use
        :param resampling: Which resampling method to use
        :param proposal: Which proposal to use, set to `auto` to let algorithm decide
        :param ess: At which level to resample
        :param need_grad: Whether we need the gradient'
        :param kwargs: Any key-worded arguments passed to `BaseFilter`
        """

        super().__init__(model, **kwargs)

        self.particles = particles
        self._th = ess

        # ===== State variables ===== #
        self._x_cur = None                          # type: torch.Tensor
        self._inds = None                           # type: torch.Tensor
        self._w_old = None                          # type: torch.Tensor

        # ===== Auxiliary variable ===== #
        self._sumaxis = -(1 + self.ssm.hidden_ndim)
        self._rsample = need_grad
        self._log_ess = log_ess

        # ===== Resampling function ===== #
        self._resampler = resampling

        # ===== Logged ESS ===== #
        self.logged_ess = TensorContainer()

        # ===== Proposal ===== #
        if proposal == 'auto':
            try:
                proposal = _PROPOSAL_MAPPING[type(self._model)]()
            except KeyError:
                proposal = Bootstrap()

        self._proposal = proposal.set_model(self._model)    # type: Proposal

    @property
    def particles(self) -> torch.Size:
        """
        Returns the number of particles.
        """

        return self._particles

    @particles.setter
    def particles(self, x: Tuple[int, int] or int):
        """
        Sets the number of particles.
        """

        self._particles = torch.Size([x]) if not isinstance(x, (tuple, list)) else torch.Size(x)

    @property
    def proposal(self) -> Proposal:
        """
        Returns the proposal.
        """

        return self._proposal

    def _resample_state(self, w: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, bool]]:
        """
        Resamples the state in accordance with the weigths.
        :param w: The weights
        :return: The indices and mask
        """

        # ===== Get the ones requiring resampling ====== #
        ess = get_ess(w) / w.shape[-1]
        mask = ess < self._th

        if self._log_ess:
            self.logged_ess.append(ess)

        # ===== Create a default array for resampling ===== #
        out = _construct_empty(w)

        # ===== Return based on if it's nested or not ===== #
        if not mask.any():
            return out, mask
        elif not isinstance(self._particles, tuple):
            return self._resampler(w), mask

        out[mask] = self._resampler(w[mask])

        return out, mask

    def set_nparallel(self, n: int):
        self._n_parallel = torch.Size([n])
        self.particles = (*self._n_parallel, *(self.particles if len(self.particles) < 2 else self.particles[1:]))

        if self._x_cur is not None:
            return self.initialize()

        return self

    def initialize(self):
        self._x_cur = self._model.hidden.i_sample(self.particles)
        self._w_old = torch.zeros(self.particles, device=self._x_cur.device)

        return self

    def predict(self, steps, aggregate: bool = True, **kwargs):
        x, y = self._model.sample_path(steps + 1, x_s=self._x_cur, **kwargs)

        if not aggregate:
            return x[1:], y[1:]

        w = normalize(self._w_old)
        wsqd = w.unsqueeze(-1)

        xm = (x * (wsqd if self.ssm.hidden_ndim > 1 else w)).sum(self._sumaxis)
        ym = (y * (wsqd if self.ssm.obs_ndim > 1 else w)).sum(-2 if self.ssm.obs_ndim > 1 else -1)

        return xm[1:], ym[1:]

    def _resample(self, inds):
        self._x_cur = self._x_cur[inds]
        self._w_old = self._w_old[inds]

        return self

    def _exchange(self, filter_, inds):
        self._x_cur[inds] = filter_._x_cur[inds]
        self._w_old[inds] = filter_._w_old[inds]

        return self

    def _reset(self):
        self.logged_ess = TensorContainer()
        return self