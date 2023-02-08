from typing import Callable, Sequence, Union

import pyro
import torch
from torch.distributions import Categorical

from ...resampling import systematic
from ..base import BaseFilter
from ..utils import batched_gather
from .proposals import Bootstrap, Proposal
from .state import ParticleFilterCorrection, ParticleFilterPrediction


class ParticleFilter(BaseFilter[ParticleFilterCorrection, ParticleFilterPrediction]):
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
        Internal initializer for :class:`ParticleFilter`.

        Args:
            model (ModelObject): see :class:`BaseFilter`.
            particles (int): number of particles to use for estimating the filter distribution.
            resampling (Callable[[torch.Tensor], torch.Tensor], optional): resampling method.. Defaults to systematic.
            proposal (Union[str, Proposal], optional): proposal distribution generator to use. Defaults to None.
            ess_threshold (float, optional):  relative "effective sample size" threshold at which to perform resampling.. Defaults to 0.9.
        """

        super().__init__(model, **kwargs)

        self._base_particles = torch.Size([particles])
        self._resample_threshold = ess_threshold * particles
        self._resampler = resampling

        if proposal is None:
            proposal = Bootstrap()

        self._proposal: Proposal = proposal

    # TODO: Invert this and rejoice
    @property
    def particles(self) -> torch.Size:
        """
        Returns the number of particles currently used by the filter. If running parallel filters, this corresponds to
        ``torch.Size([number of parallel filters, number of particles])``, else ``torch.Size([number of particles])``.
        """

        return torch.Size(
            [
                *self._base_particles,
                *self.batch_shape,
            ]
        )

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
            factor (int): the factor to increase the particles with.
        """

        self._base_particles = torch.Size([int(factor * self._base_particles[0])])
        self._resample_threshold *= factor
    
    def initialize_model(self, context):
        super().initialize_model(context)
        self._proposal.set_model(self._model)

    def initialize(self):
        assert self._model is not None, "Model has not been initialized!"
        
        self._proposal.set_model(self.ssm)
        x = self._model.hidden.initial_sample(self.particles)

        device = x.value.device

        weights = torch.zeros(self.particles, device=device)
        prev_inds = torch.arange(weights.shape[0], device=device)

        if self.batch_shape:
            prev_inds = prev_inds.unsqueeze(-1).expand(self.particles)

        ll = torch.zeros(self.batch_shape, device=device)

        return ParticleFilterCorrection(x, weights, ll, prev_inds)

    def _do_sample_ffbs(self, states: Sequence[ParticleFilterCorrection]):
        dim = 0
        res = [batched_gather(states[-1].timeseries_state.value, self._resampler(states[-1].weights), dim=dim)]

        for state in reversed(states[:-1]):
            density = self.ssm.hidden.build_density(state.timeseries_state)

            # TODO: Something is wrong here, fix
            w_state = density.log_prob(res[-1].unsqueeze(1))
            w = state.weights.unsqueeze(0) + w_state

            if self.batch_shape:
                w = w.moveaxis(1, 2)

            indices = Categorical(logits=w).sample()

            if self.ssm.hidden.n_dim > 0:
                indices = indices.unsqueeze(-1).expand(self.particles + self.ssm.hidden.event_shape)

            resampled = state.timeseries_state.value.gather(0, indices)
            res.append(resampled)

        return torch.stack(res[::-1], dim=0)

    def _do_sample_fl(self, states: Sequence[ParticleFilterCorrection]):
        reversed_states = reversed(states)

        latest_state = next(reversed_states)
        result = (latest_state.timeseries_state.value,)
        prev_inds = torch.arange(0, self._base_particles[0], device=result[-1].device)

        if self.batch_shape:
            prev_inds = prev_inds.unsqueeze(-1).expand(self.particles)

        dim = 0
        for s in reversed_states:
            prev_inds = batched_gather(latest_state.previous_indices, prev_inds, dim=dim)
            result += (batched_gather(s.timeseries_state.value, prev_inds, dim=dim),)
            latest_state = s

        return torch.stack(result[::-1], dim=0)

    # NB: Discrepancy between shape of filter means and here
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

        hidden_density = self.ssm.hidden.build_density(x_tm1)
        obs_density = self.ssm.build_density(x_t)
        init_density = self.ssm.hidden.initial_distribution

        shape = (y.shape[0], *(len(obs_density.batch_shape[1:]) * [1]), *y.shape[1:])
        y_ = y.view(shape)
        tot_prob = (
            hidden_density.log_prob(smoothed[1:]).sum(0)
            + obs_density.log_prob(y_).sum(0)
            + init_density.log_prob(smoothed[0])
        )

        pyro_lib.factor("log_prob", tot_prob.mean(0))

    def do_sample_pyro(self, y: torch.Tensor, pyro_lib: pyro, method="ffbs"):
        """
        Performs a smoothing procedure in which we acquire the log-likelihood for `pyro` to target.

        This is an experimental feature, as the author needs to find theoretical justifications for this approach.
        Currently does not work with vectorized inference.

        Args:
            y (torch.Tensor): observations to use when filtering.
            pyro_lib (pyro): pyro library.
            method (str, optional): method to use when constructing a target log-likelihood.. Defaults to "ffbs".
        """

        lower_method = method.lower()
        if lower_method == "ffbs":
            self._do_pyro_ffbs(y, pyro_lib)
        else:
            raise NotImplementedError(f"Currently do not support '{method}'!")
