from abc import ABC
from typing import Callable, Sequence, Union

import pyro
import torch
from torch.distributions import Categorical

from ...resampling import systematic
from ..base import BaseFilter
from ..utils import batched_gather
from .proposals import Bootstrap, Proposal
from .state import ParticleFilterCorrection, ParticleFilterPrediction


class ParticleFilter(BaseFilter[ParticleFilterCorrection, ParticleFilterPrediction], ABC):
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
        **kwargs,
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

        self._proposal: Proposal = proposal

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

        self._base_particles = torch.Size([int(factor * self._base_particles[0])])

    def initialize(self):
        assert self._model is not None, "Model has not been initialized!"
            
        self._proposal.set_model(self._model)
        x = self._model.hidden.initial_sample(self.particles)

        device = x.value.device

        w = torch.zeros(self.particles, device=device)
        prev_inds = torch.ones(w.shape, dtype=torch.int, device=device) * torch.arange(w.shape[-1], device=device)
        ll = torch.zeros(self.batch_shape, device=device)

        return ParticleFilterCorrection(x, w, ll, prev_inds)

    def _do_sample_ffbs(self, states: Sequence[ParticleFilterCorrection]):
        state_dim = -(1 + self.ssm.hidden.n_dim)
        dim = len(self.batch_shape)

        res = [batched_gather(states[-1].timeseries_state.value, self._resampler(states[-1].weights), dim=dim)]

        for state in reversed(states[:-1]):
            density = self.ssm.hidden.build_density(state.timeseries_state)

            # TODO: Last transpose might not be necessary, figure it out
            w_state = density.log_prob(res[-1].unsqueeze(0).transpose(0, state_dim))

            if self.batch_shape:
                w_state = w_state.transpose(0, 1)

            w = state.weights.unsqueeze(-2) + w_state

            cat = Categorical(logits=w)
            res.append(batched_gather(state.timeseries_state.value, cat.sample(), dim=dim))

        return torch.stack(res[::-1], dim=0)

    def _do_sample_fl(self, states: Sequence[ParticleFilterCorrection]):
        reversed_states = reversed(states)

        latest_state = next(reversed_states)
        result = (latest_state.timeseries_state.value,)
        prev_inds = torch.ones_like(latest_state.previous_indices).cumsum(dim=-1) - 1

        dim = len(self.batch_shape)
        for s in reversed_states:
            prev_inds = batched_gather(latest_state.previous_indices, prev_inds, dim=dim)
            result += (batched_gather(s.timeseries_state.value, prev_inds, dim=dim),)
            latest_state = s

        return torch.stack(result[::-1], dim=0)

    def smooth(self, states, method="ffbs") -> torch.Tensor:
        lower_method = method.lower()
        if lower_method == "ffbs":
            return self._do_sample_ffbs(states)
        
        if method == "fl":
            return self._do_sample_fl(states)

        raise NotImplementedError(f"Currently do not support '{method}'!")

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
            record_intermediary_states=self._record_intermediary,
        )

        res.set_batch_shape(self.batch_shape)

        return res

    def _do_pyro_ffbs(self, y: torch.Tensor, pyro_lib: pyro):
        """
        Performs the `Forward Filtering Backward Sampling` procedure in order to obtain the log-likelihood w.r.t. to the
        parameters, and then registers the resulting log-likelihood as a factor for `pyro` to use when optimizing.
        """

        assert self.record_states is True, "Must record all states! Set `record_states=True` when initializing"

        with torch.no_grad():
            result = self.batch_filter(y, bar=False)
            smoothed = self.smooth(result.states, method="ffbs")

            time_indexes = torch.stack([s.get_timeseries_state().time_index for s in result.states])

            init_state = self.ssm.hidden.initial_sample()
            x_tm1 = init_state.propagate_from(smoothed[:-1], time_increment=time_indexes[:-1])
            x_t = init_state.propagate_from(
                smoothed[1 :: self.ssm.observe_every_step],
                time_increment=time_indexes[1 :: self.ssm.observe_every_step],
            )

        hidden_dens = self.ssm.hidden.build_density(x_tm1)
        obs_dens = self.ssm.build_density(x_t)
        init_dens = self.ssm.hidden.initial_distribution
        init_dens = self.ssm.hidden.initial_distribution

        shape = (y.shape[0], *(len(obs_dens.batch_shape[1:]) * [1]), *obs_dens.event_shape)
        y_ = y.view(shape)
        tot_prob = (
            hidden_dens.log_prob(smoothed[1:]).sum(0) + obs_dens.log_prob(y_).sum(0) + init_dens.log_prob(smoothed[0])
        )

        pyro_lib.factor("log_prob", tot_prob.mean(-1))

    def do_sample_pyro(self, y: torch.Tensor, pyro_lib: pyro, method="ffbs"):
        """
        Performs a filtering procedure in which we acquire the log-likelihood for `pyro` to target.

        This is an experimental feature, as the author needs to find theoretical justifications for this approach.
        Currently does not work with vectorized inference.

        Args:
            y: observations to use when filtering.
            pyro_lib: pyro library.
            method: method to use when constructing a target log-likelihood.
        """

        lower_method = method.lower()
        if lower_method == "ffbs":
            self._do_pyro_ffbs(y, pyro_lib)
        else:
            raise NotImplementedError(f"Currently do not support '{method}'!")
