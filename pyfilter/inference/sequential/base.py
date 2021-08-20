from abc import ABC
import torch
from .state import SequentialAlgorithmState
from ..logging import TQDMWrapper
from ..base import BaseFilterAlgorithm
from ..utils import sample_model
from ...utils import normalize
from ...filters import ParticleFilter, FilterResult


class SequentialFilteringAlgorithm(BaseFilterAlgorithm, ABC):
    """
    Base class for sequential algorithms using filters in order to approximate the log likelihood.
    """

    def update(self, y: torch.Tensor, state: SequentialAlgorithmState) -> SequentialAlgorithmState:
        """
        Performs an update using a single observation `y`.
        """

        raise NotImplementedError()

    def fit(self, y, logging=None, **kwargs) -> SequentialAlgorithmState:
        logging = logging or TQDMWrapper()
        logging.initialize(self, y.shape[0])

        try:
            state = self.initialize()
            for i, yt in enumerate(y):
                state = self.update(yt, state)
                logging.do_log(i, state)

            return state

        except Exception as e:
            raise e
        finally:
            logging.close()


class SequentialParticleAlgorithm(SequentialFilteringAlgorithm, ABC):
    """
    Base class for sequential algorithms using particles to approximate the distribution of the parameters.
    """

    def __init__(self, filter_, particles: int):
        super().__init__(filter_)

        self.register_buffer("_particles", torch.tensor(particles, dtype=torch.int))
        self.filter.set_nparallel(particles)

    @property
    def particles(self) -> torch.Size:
        return torch.Size([self._particles])

    def sample_params(self):
        shape = torch.Size((*self.particles, 1)) if isinstance(self.filter, ParticleFilter) else self.particles
        sample_model(self.filter.ssm, shape)

    def initialize(self) -> SequentialAlgorithmState:
        self.sample_params()

        init_state = self.filter.initialize()
        init_weights = torch.zeros(self.particles, device=init_state.get_loglikelihood().device)

        return SequentialAlgorithmState(init_weights, FilterResult(init_state, record_states=self.filter.record_states))

    def predict(self, steps, state: SequentialAlgorithmState, aggregate=True, **kwargs):
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
        Algorithm combining two instances of `SequentialParticleAlgorithm`. One such example is the `NESSMC2`, where we
        utilize the `SMC2` for an arbitrary chunk of the data first, and then switch to the `NESS` algorithm.

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
        self, first: SequentialParticleAlgorithm, second: SequentialParticleAlgorithm, state: SequentialAlgorithmState
    ) -> SequentialAlgorithmState:
        raise NotImplementedError()

    def initialize(self):
        return self._first.initialize()

    def update(self, y: torch.Tensor, state):
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
