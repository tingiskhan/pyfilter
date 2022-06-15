from abc import ABC
from typing import Tuple, Union, Callable
import torch
from torch.distributions import Categorical

from ..base import BaseFilter
from ...resampling import systematic
from .proposals import Bootstrap, Proposal
from .state import ParticleFilterState
from ..utils import batched_gather


class ParticleFilter(BaseFilter, ABC):
    """
    Abstract base class for particle filters.
    """

    def __init__(
        self,
        model,
        particles: int,
        resampling: Callable[[torch.Tensor], torch.Tensor] = systematic,
        proposal: Union[str, Proposal] = None,
        ess_threshold=0.9,
        **kwargs
    ):
        """
        Initializes the :class:`ParticleFilter` class.

        Args:
            model: see base.
            particles: the number of particles to use for estimating the filter distribution.
            resampling: the resampling method. Takes as input the log weights and returns mask.
            proposal: the proposal distribution generator to use.
            ess_threshold: the relative "effective sample size" threshold at which to perform resampling. Not relevant
                for ``APF`` as resampling is always performed.
        """

        super().__init__(model, **kwargs)

        self._base_particles = torch.Size([particles])
        self._resample_threshold = ess_threshold
        self._resampler = resampling

        if proposal is None:
            proposal = Bootstrap()

        self._proposal = proposal.set_model(self._model)  # type: Proposal

    @property
    def particles(self) -> torch.Size:
        """
        Returns the number of particles currently used by the filter. If running parallel filters, this corresponds to
        ``torch.Size([number of parallel filters, number of particles])``, else ``torch.Size([number of particles])``.
        """

        return torch.Size([*self.batch_shape, *self._base_particles])

    @property
    def proposal(self) -> Proposal:
        """
        Returns the proposal used by the filter.
        """

        return self._proposal

    def increase_particles(self, factor: int):
        """
        Increases the particle count by ``factor``.

        Args:
            factor: the factor to increase the particles with.

        """

        self._base_particles = torch.Size([factor * self._base_particles[0]])

    def initialize(self) -> ParticleFilterState:
        x = self._model.hidden.initial_sample(self.particles)

        device = x.values.device

        w = torch.zeros(self.particles, device=device)
        prev_inds = torch.ones(w.shape, dtype=torch.long, device=device) * torch.arange(w.shape[-1], device=device)
        ll = torch.zeros(self.batch_shape, device=device)

        return ParticleFilterState(x, w, ll, prev_inds)

    def smooth(self, states: Tuple[ParticleFilterState]) -> torch.Tensor:
        offset = -(2 + self.ssm.hidden.n_dim)

        dim_to_unsqueeze = -2
        for p in self.ssm.hidden.parameters():
            if p.dim() > 0:
                p.unsqueeze_(dim_to_unsqueeze)

        dim = len(self.batch_shape)

        res = [batched_gather(states[-1].x.values, self._resampler(states[-1].w), dim=dim)]
        for state in reversed(states[:-1]):
            temp_state = state.x.copy(values=state.x.values.unsqueeze(offset))
            density = self.ssm.hidden.build_density(temp_state)

            w = state.w.unsqueeze(-2) + density.log_prob(res[-1].unsqueeze(offset + 1))

            cat = Categorical(logits=w)
            res.append(batched_gather(state.x.values, cat.sample(), dim=dim))

        for p in self.ssm.hidden.parameters():
            if p.dim() > 0:
                p.squeeze_(dim_to_unsqueeze)

        return torch.stack(res[::-1], dim=0)

    def copy(self):
        res = type(self)(
            model=self._model_builder,
            particles=self._base_particles[0],
            resampling=self._resampler,
            proposal=self._proposal.copy(),
            ess_threshold=self._resample_threshold,
            record_states=self.record_states,
            record_moments=self.record_moments,
            nan_strategy=self._nan_strategy,
            record_intermediary_states=self._record_intermediary
        )

        res.set_batch_shape(self.batch_shape)

        return res

