from abc import ABC
import torch
from typing import Dict, Any
from .state import SequentialAlgorithmState
from ..logging import TQDMWrapper
from ..base import BaseAlgorithm


class SequentialParticleAlgorithm(BaseAlgorithm, ABC):
    """
    Abstract base class for sequential algorithms using filters in order to approximate the log likelihood.
    """

    def __init__(self, filter_, num_particles: int):
        """
        Initializes the :class:`SequentialParticleAlgorithm` class.

        Args:
            filter_: See base.
            num_particles: The number of particles to use for approximating the parameter posteriors.
        """

        super().__init__(filter_)

        self.particles = torch.Size([num_particles])
        self.filter.set_batch_shape(self.particles)

    def initialize(self) -> SequentialAlgorithmState:
        """
        Initializes the algorithm by returning an ``SequentialAlgorithmState``.
        """

        init_state = self.filter.initialize()
        init_weights = torch.zeros(self.particles, device=init_state.get_loglikelihood().device)

        return SequentialAlgorithmState(init_weights, self.filter.initialize_with_result(init_state))

    def step(self, y: torch.Tensor, state: SequentialAlgorithmState) -> SequentialAlgorithmState:
        """
        Updates the algorithm and filter state given the latest observation ``y``.

        Args:
            y: The latest observation.
            state: The previous state of the algorithm.

        Returns:
            The updated state of the algorithm.
        """

        raise NotImplementedError()

    def fit(self, y, logging=None, **kwargs) -> SequentialAlgorithmState:
        logging = logging or TQDMWrapper()
        logging.initialize(self, y.shape[0])

        try:
            state = self.initialize()
            for i, yt in enumerate(y):
                state = self.step(yt, state)
                logging.do_log(i, state)

            return state

        except Exception as e:
            raise e
        finally:
            logging.teardown()

    def predict(self, steps, state: SequentialAlgorithmState, aggregate=True, **kwargs):
        px, py = self.filter.predict_path(state.filter_state.latest_state, steps, aggregate=aggregate, **kwargs)

        if not aggregate:
            return px, py

        w = state.normalized_weights()
        w_unsqueezed = w.unsqueeze(-1)

        x_m = (px * (w_unsqueezed if self.filter.ssm.hidden.n_dim > 0 else w)).sum(1)
        y_m = (py * (w_unsqueezed if self.filter.ssm.observable.n_dim > 0 else w)).sum(1)

        return x_m, y_m


class CombinedSequentialParticleAlgorithm(SequentialParticleAlgorithm, ABC):
    """
    Algorithm combining two instances of ``SequentialParticleAlgorithm``, where we let one of them target a
    chronological subset of the data, and the other the remaining points.

    One such example is the ``NESSMC2``, where we first utilize the costly but exact ``SMC2`` algorithm, and then switch
    to the ``NESS`` algorithm which is a pure online algorithm, but with slower convergence than ``SMC2``.
    """

    def __init__(self, filter_, particles, switch: int, first_kw: Dict[str, Any], second_kw: Dict[str, Any]):
        """
        Initializes the ``CombinedSequentialParticleAlgorithm`` class.

        Args:
            filter_: See base.
            particles: See base.
            switch: The number of observations to have parsed before switching algorithms.
            first_kw: Kwargs sent to ``.make_first(...)``.
            second_kw: Kwargs sent to ``.make_second(...)``.
        """

        super().__init__(filter_, particles)

        self._first = self.make_first(filter_, particles, **(first_kw or dict()))
        self._second = self.make_second(filter_, particles, **(second_kw or dict()))

        self._when_to_switch = switch
        self._is_switched = False
        self._num_iterations = 0

    def make_first(self, filter_, particles, **kwargs) -> SequentialParticleAlgorithm:
        """
        Creates the algorithm to be used for the first part of the data.

        Args:
            filter_: See ``__init__``.
            particles: See ``__init__``.
            kwargs: Corresponds to ``first_kw`` of ``__init__``.

        Returns:
            Instance of algorithm to be used for first part of the data.
        """

        raise NotImplementedError()

    def make_second(self, filter_, particles, **kwargs) -> SequentialParticleAlgorithm:
        """
        See ``.make_first(...)`` but replace `first` with `second`.
        """

        raise NotImplementedError()

    def do_on_switch(
        self, first: SequentialParticleAlgorithm, second: SequentialParticleAlgorithm, state: SequentialAlgorithmState
    ) -> SequentialAlgorithmState:
        raise NotImplementedError()

    def initialize(self):
        return self._first.initialize()

    def step(self, y: torch.Tensor, state):
        self._num_iterations += 1

        if not self._is_switched:
            if self._num_iterations <= self._when_to_switch:
                return self._first.step(y, state)

            self._is_switched = True
            state = self.do_on_switch(self._first, self._second, state)

        return self._second.step(y, state)

    def predict(self, steps, state, aggregate=True, **kwargs):
        if not self._is_switched:
            return self._first.predict(steps, state, aggregate=aggregate, **kwargs)

        return self._second.predict(steps, state, aggregate=aggregate, **kwargs)
