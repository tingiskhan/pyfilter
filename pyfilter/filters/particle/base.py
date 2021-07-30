from abc import ABC
from typing import Tuple, Union, Callable
import torch
from torch.distributions import Categorical
from ..base import BaseFilter
from ...resampling import systematic
from ...timeseries import LinearGaussianObservations as LGO
from .proposals import Bootstrap, Proposal, LinearGaussianObservations
from ...utils import get_ess, normalize, choose
from ..utils import _construct_empty
from .state import ParticleState


_PROPOSAL_MAPPING = {LGO.__name__: LinearGaussianObservations}


class ParticleFilter(BaseFilter, ABC):
    def __init__(
        self,
        model,
        particles: int,
        resampling: Callable[[torch.Tensor], torch.Tensor] = systematic,
        proposal: Union[str, Proposal] = "auto",
        ess=0.9,
        **kwargs
    ):
        """
        Implements the base functionality of a particle filter.

        :param particles: How many particles to use
        :param resampling: Which resampling method to use
        :param proposal: Which proposal to use, set to `auto` to let algorithm decide
        :param ess: At which level to resample
        :param kwargs: Any key-worded arguments passed to `BaseFilter`
        """

        super().__init__(model, **kwargs)

        self.register_buffer("_particles", torch.tensor(particles, dtype=torch.int))
        self._resample_threshold = ess
        self._resampler = resampling

        if proposal == "auto":
            proposal = _PROPOSAL_MAPPING.get(self._model.__class__.__name__, Bootstrap)()

        self._proposal = proposal.set_model(self._model)  # type: Proposal

    @property
    def particles(self) -> torch.Size:
        return torch.Size([self._particles] if self._particles.dim() == 0 else self._particles)

    @property
    def proposal(self) -> Proposal:
        return self._proposal

    def _resample_state(self, w: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, bool]]:
        ess = get_ess(w) / w.shape[-1]
        mask = ess < self._resample_threshold

        out = _construct_empty(w)

        if not mask.any():
            return out, mask
        elif not isinstance(self._particles, tuple):
            return self._resampler(w), mask

        out[mask] = self._resampler(w[mask])

        return out, mask

    def set_nparallel(self, num_filters: int):
        self._n_parallel = torch.tensor(num_filters)
        self._particles = torch.tensor(
            (*self.n_parallel, *(self.particles if len(self.particles) < 2 else self.particles[1:])), dtype=torch.int
        )

        return self

    def initialize(self) -> ParticleState:
        x = self._model.hidden.initial_sample(self.particles)
        w = torch.zeros(self.particles, device=x.device)
        prev_inds = torch.ones(w.shape, dtype=torch.long, device=x.device) * torch.arange(w.shape[-1], device=x.device)

        return ParticleState(x, w, torch.zeros(self.n_parallel, device=x.device), prev_inds)

    def predict(self, state: ParticleState, steps, aggregate: bool = True, **kwargs):
        x, y = self._model.sample_path(steps, x_s=state.x, **kwargs)

        x = x[1:]
        if not aggregate:
            return x, y

        w = normalize(state.w)
        squeezed_w = w.unsqueeze(-1)

        sum_axis = -(1 + self.ssm.hidden.n_dim)

        obs_ndim = self.ssm.observable.n_dim
        x_mean = (x * (squeezed_w if self.ssm.hidden.n_dim > 0 else w)).sum(sum_axis)
        y_mean = (y * (squeezed_w if obs_ndim > 0 else w)).sum(-2 if obs_ndim > 0 else -1)

        return x_mean, y_mean

    # TODO: Might not work when we have parameters of wrong size...?
    def smooth(self, states: Tuple[ParticleState]) -> torch.Tensor:
        hidden_copy = self.ssm.hidden.copy()
        offset = -(2 + self.ssm.hidden.n_dim)

        for p in hidden_copy.parameters():
            p.unsqueeze_(-2)

        res = [choose(states[-1].x.values, self._resampler(states[-1].w))]
        for state in reversed(states[:-1]):
            temp_state = state.x.copy(values=state.x.values.unsqueeze(offset))
            density = hidden_copy.build_density(temp_state)

            w = state.w.unsqueeze(-2) + density.log_prob(res[-1].unsqueeze(offset + 1))

            cat = Categorical(logits=w)
            res.append(choose(state.x.values, cat.sample()))

        return torch.stack(res[::-1], dim=0)
