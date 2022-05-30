from abc import ABC
from typing import Tuple, Union, Callable
import torch
from torch.distributions import Categorical

from ..base import BaseFilter
from ...resampling import systematic
from .proposals import Bootstrap, Proposal
from ...utils import choose
from .state import ParticleFilterState, ParticleFilterPrediction


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
        Initializes the ``ParticleFilter`` class.

        Args:
            model: See base.
            particles: The number of particles to use for estimating the filter distribution.
            resampling: The resampling method. Takes as input the log weights and returns indices.
            proposal: The proposal distribution generator to use.
            ess_threshold: The relative "effective sample size" threshold at which to perform resampling. Not relevant
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

    def initialize(self) -> ParticleFilterState:
        x = self._model.hidden.initial_sample(self.particles)

        device = x.values.device

        w = torch.zeros(self.particles, device=device)
        prev_inds = torch.ones(w.shape, dtype=torch.long, device=device) * torch.arange(w.shape[-1], device=device)
        ll = torch.zeros(self.batch_shape, device=device)

        return ParticleFilterState(x, w, ll, prev_inds)

    def predict_path(self, state: ParticleFilterState, steps, aggregate: bool = True, **kwargs):
        x, y = self._model.sample_path(steps, x_s=state.x, **kwargs)

        x = x[1:]
        if not aggregate:
            return x, y

        w = state.normalized_weights()
        w_unsqueezed = w.unsqueeze(-1)

        sum_axis = -(1 + self.ssm.hidden.n_dim)

        obs_ndim = self.ssm.observable.n_dim
        x_mean = (x * (w_unsqueezed if self.ssm.hidden.n_dim > 0 else w)).sum(sum_axis)
        y_mean = (y * (w_unsqueezed if obs_ndim > 0 else w)).sum(-2 if obs_ndim > 0 else -1)

        return x_mean, y_mean

    def smooth(self, states: Tuple[ParticleFilterState]) -> torch.Tensor:
        hidden_copy = self.ssm.hidden.copy()
        offset = -(2 + self.ssm.hidden.n_dim)

        for p in hidden_copy.parameters():
            if p.dim() > 0:
                p.unsqueeze_(-2)

        res = [choose(states[-1].x.values, self._resampler(states[-1].w))]
        for state in reversed(states[:-1]):
            temp_state = state.x.copy(values=state.x.values.unsqueeze(offset))
            density = hidden_copy.build_density(temp_state)

            w = state.w.unsqueeze(-2) + density.log_prob(res[-1].unsqueeze(offset + 1))

            cat = Categorical(logits=w)
            res.append(choose(state.x.values, cat.sample()))

        return torch.stack(res[::-1], dim=0)

    def _get_observation_dist_from_prediction(self, prediction: ParticleFilterPrediction):
        raise NotImplementedError()
