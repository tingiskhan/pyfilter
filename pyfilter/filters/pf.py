from .base import BaseFilter
from abc import ABC
from ..resampling import systematic
from ..timeseries import LinearGaussianObservations as LGO
from .proposals import Bootstrap, Proposal, LinearGaussianObservations
import torch
from ..utils import get_ess, normalize, choose
from .utils import _construct_empty
from typing import Tuple, Union, Iterable
from .state import ParticleState
from torch.distributions import Categorical


_PROPOSAL_MAPPING = {LGO.__name__: LinearGaussianObservations}


class ParticleFilter(BaseFilter, ABC):
    def __init__(
        self,
        model,
        particles: int,
        resampling=systematic,
        proposal: Union[str, Proposal] = "auto",
        ess=0.9,
        need_grad=False,
        **kwargs
    ):
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

        self.register_buffer("_particles", torch.tensor(particles, dtype=torch.int))
        self._th = ess

        self._sumaxis = -(1 + self.ssm.hidden_ndim)
        self._rsample = need_grad

        self._resampler = resampling

        if proposal == "auto":
            try:
                proposal = _PROPOSAL_MAPPING[self._model.__class__.__name__]()
            except KeyError:
                proposal = Bootstrap()

        self._proposal = proposal.set_model(self._model)  # type: Proposal

    @property
    def particles(self) -> torch.Size:
        return torch.Size([self._particles] if self._particles.dim() == 0 else self._particles)

    @property
    def proposal(self) -> Proposal:
        return self._proposal

    def _resample_state(self, w: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, bool]]:
        ess = get_ess(w) / w.shape[-1]
        mask = ess < self._th

        out = _construct_empty(w)

        if not mask.any():
            return out, mask
        elif not isinstance(self._particles, tuple):
            return self._resampler(w), mask

        out[mask] = self._resampler(w[mask])

        return out, mask

    def set_nparallel(self, n: int):
        self._n_parallel = torch.tensor(n)
        self._particles = torch.tensor(
            (*self.n_parallel, *(self.particles if len(self.particles) < 2 else self.particles[1:])), dtype=torch.int
        )

        return self

    def initialize(self) -> ParticleState:
        x = self._model.hidden.i_sample(self.particles)
        w = torch.zeros(self.particles, device=x.device)
        prev_inds = torch.ones_like(w) * torch.arange(w.shape[-1], device=x.device)

        return ParticleState(x, w, torch.zeros(self.n_parallel, device=x.device), prev_inds)

    def predict(self, state: ParticleState, steps, aggregate: bool = True, **kwargs):
        x, y = self._model.sample_path(steps + 1, x_s=state.x, **kwargs)

        if not aggregate:
            return x[1:], y[1:]

        w = normalize(state.w)
        wsqd = w.unsqueeze(-1)

        xm = (x * (wsqd if self.ssm.hidden_ndim > 1 else w)).sum(self._sumaxis)
        ym = (y * (wsqd if self.ssm.obs_ndim > 1 else w)).sum(-2 if self.ssm.obs_ndim > 1 else -1)

        return xm[1:], ym[1:]

    # FIXME: Fix this, not working currently I think
    def smooth(self, states: Iterable[ParticleState]):
        hidden_copy = self.ssm.hidden.copy((*self.n_parallel, 1, 1))
        offset = -(2 + self.ssm.hidden_ndim)

        res = [choose(states[-1].x, self._resampler(states[-1].w))]
        reverse = states[::-1]
        for state in reverse[1:]:
            w = state.w.unsqueeze(-2) + hidden_copy.log_prob(res[-1].unsqueeze(offset), state.x.unsqueeze(offset + 1))

            cat = Categorical(normalize(w))
            res.append(choose(state.x, cat.sample()))

        return torch.stack(res[::-1], dim=0)
