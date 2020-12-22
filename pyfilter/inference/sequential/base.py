from abc import ABC
from ...filters import ParticleFilter, utils as u, FilterResult
import torch
from ...utils import normalize
from ..base import BaseFilterAlgorithm
from .state import FilteringAlgorithmState


class SequentialFilteringAlgorithm(BaseFilterAlgorithm, ABC):
    def _update(self, y: torch.Tensor, state: FilteringAlgorithmState) -> FilteringAlgorithmState:
        raise NotImplementedError()

    @u.enforce_tensor
    def update(self, y: torch.Tensor, state: FilteringAlgorithmState) -> FilteringAlgorithmState:
        """
        Performs an update using a single observation `y`.
        :param y: The observation
        :param state: The previous state
        :return: Self
        """

        return self._update(y, state)

    def _fit(self, y, logging_wrapper=None, **kwargs) -> FilteringAlgorithmState:
        logging_wrapper.set_num_iter(y.shape[0])
        try:
            state = self.initialize()
            for i, yt in enumerate(y):
                state = self.update(yt, state)
                logging_wrapper.do_log(i, self, y)

        except Exception as e:
            logging_wrapper.close()
            raise e

        logging_wrapper.close()

        return state


class SequentialParticleAlgorithm(SequentialFilteringAlgorithm, ABC):
    def __init__(self, filter_, particles: int):
        super().__init__(filter_)

        self.register_buffer("_particles", torch.tensor(particles, dtype=torch.int))
        self.filter.set_nparallel(particles)

    @property
    def particles(self) -> torch.Size:
        return torch.Size([self._particles])

    def sample_params(self):
        shape = torch.Size((*self.particles, 1)) if isinstance(self.filter, ParticleFilter) else self.particles
        self.filter.ssm.sample_params(shape)

    def initialize(self) -> FilteringAlgorithmState:
        self.sample_params()

        init_state = self.filter.initialize()
        init_weights = torch.zeros(self.particles, device=init_state.get_loglikelihood().device)

        return FilteringAlgorithmState(init_weights, FilterResult(init_state))

    def predict(self, steps, state: FilteringAlgorithmState, aggregate=True, **kwargs):
        px, py = self.filter.predict(state.filter_state.latest_state, steps, aggregate=aggregate, **kwargs)

        if not aggregate:
            return px, py

        w = normalize(state.w)
        wsqd = w.unsqueeze(-1)

        xm = (px * (wsqd if self.filter.ssm.hidden_ndim > 1 else w)).sum(1)
        ym = (py * (wsqd if self.filter.ssm.obs_ndim > 1 else w)).sum(1)

        return xm, ym


class CombinedSequentialParticleAlgorithm(SequentialParticleAlgorithm, ABC):
    def __init__(self, filter_, particles, switch: int, first_kw, second_kw):
        """
        Algorithm combining two other algorithms.
        :param switch: After how many observations to perform switch
        """

        super().__init__(filter_, particles)
        self._first = self.make_first(filter_, particles, **(first_kw or dict()))
        self._second = self.make_second(filter_, particles, **(second_kw or dict()))
        self._when_to_switch = switch
        self._is_switched = torch.tensor(False, dtype=torch.bool)
        self._num_iters = 0

    def make_first(self, filter_, particles, **kwargs) -> SequentialParticleAlgorithm:
        raise NotImplementedError()

    def make_second(self, filter_, particles, **kwargs) -> SequentialParticleAlgorithm:
        raise NotImplementedError()

    def do_on_switch(
        self, first: SequentialParticleAlgorithm, second: SequentialParticleAlgorithm, state: FilteringAlgorithmState
    ) -> FilteringAlgorithmState:
        raise NotImplementedError()

    def initialize(self):
        return self._first.initialize()

    def _update(self, y: torch.Tensor, state):
        self._num_iters += 1

        if not self._is_switched:
            if self._num_iters <= self._when_to_switch:
                return self._first.update(y, state)

            self._is_switched = True
            state = self.do_on_switch(self._first, self._second, state)

        return self._second.update(y, state)

    def predict(self, steps, state, aggregate=True, **kwargs):
        if not self._is_switched:
            return self._first.predict(steps, state, aggregate=aggregate, **kwargs)

        return self._second.predict(steps, state, aggregate=aggregate, **kwargs)
