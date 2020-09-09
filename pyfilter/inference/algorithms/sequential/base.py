from abc import ABC
from ....filters import ParticleFilter, utils as u, BaseState
from ....module import TensorContainer
import torch
from ....utils import normalize
from ..base import BaseFilterAlgorithm


class SequentialFilteringAlgorithm(BaseFilterAlgorithm, ABC):
    """
    Algorithm for sequential inference.
    """

    def _update(self, y: torch.Tensor, state: BaseState) -> BaseState:
        raise NotImplementedError()

    @u.enforce_tensor
    def update(self, y: torch.Tensor, state: BaseState) -> BaseState:
        """
        Performs an update using a single observation `y`.
        :param y: The observation
        :param state: The previous state
        :return: Self
        """

        return self._update(y, state)

    def _fit(self, y, logging_wrapper=None, **kwargs):
        logging_wrapper.set_num_iter(y.shape[0])

        state = self.filter.initialize()
        for i, yt in enumerate(y):
            state = self.update(yt, state)
            logging_wrapper.do_log(i, self, y)

        return self


class SequentialParticleAlgorithm(SequentialFilteringAlgorithm, ABC):
    def __init__(self, filter_, particles: int):
        """
        Implements a base class for sequential particle inference.
        :param particles: The number of particles to use
        """

        super().__init__(filter_)

        # ===== Weights ===== #
        self._w_rec = None  # type: torch.Tensor

        # ===== ESS related ===== #
        self._logged_ess = TensorContainer()
        self.particles = particles

    @property
    def particles(self) -> torch.Size:
        """
        Returns the number of particles.
        """

        return self._particles

    @particles.setter
    def particles(self, x: int):
        """
        Sets the particles.
        """

        self._particles = torch.Size([x])

    def viewify_params(self):
        shape = torch.Size((*self.particles, 1)) if isinstance(self.filter, ParticleFilter) else self.particles
        self.filter.viewify_params(shape)

        return self

    def initialize(self) -> BaseFilterAlgorithm:
        """
        Overwrites the initialization.
        :return: Self
        """

        self.filter.set_nparallel(*self.particles)
        self.filter.ssm.sample_params(self.particles)
        self._w_rec = torch.zeros(self.particles, device=self.filter._dummy.device)

        self.viewify_params()

        return self

    @property
    def logged_ess(self) -> torch.Tensor:
        """
        Returns the logged ESS.
        """

        return torch.stack(self._logged_ess.tensors)

    def predict(self, steps, state: BaseState, aggregate=True, **kwargs):
        px, py = self.filter.predict(state, steps, aggregate=aggregate, **kwargs)

        if not aggregate:
            return px, py

        w = normalize(self._w_rec)
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

    def make_first(self, filter_, particles, **kwargs) -> SequentialParticleAlgorithm:
        raise NotImplementedError()

    def make_second(self, filter_, particles, **kwargs) -> SequentialParticleAlgorithm:
        raise NotImplementedError()

    def do_on_switch(self, first: SequentialParticleAlgorithm, second: SequentialParticleAlgorithm, state: BaseState):
        raise NotImplementedError()

    def initialize(self) -> BaseFilterAlgorithm:
        self._first.initialize()
        return self

    def modules(self):
        return {
            '_first': self._first,
            '_second': self._second
        }

    @property
    def logged_ess(self):
        return torch.cat((self._first.logged_ess, self._second.logged_ess))

    @property
    def _w_rec(self):
        if self._is_switched:
            return self._first._w_rec

        return self._second._w_rec

    @_w_rec.setter
    def _w_rec(self, x):
        return

    def _update(self, y: torch.Tensor, state):
        if not self._is_switched:
            if len(self._first._logged_ess) < self._when_to_switch:
                return self._first.update(y, state)

            self._is_switched = True
            self.do_on_switch(self._first, self._second, state)

        return self._second.update(y, state)

    def predict(self, steps, state, aggregate=True, **kwargs):
        if not self._is_switched:
            return self._first.predict(steps, state, aggregate=aggregate, **kwargs)

        return self._second.predict(steps, state, aggregate=aggregate, **kwargs)