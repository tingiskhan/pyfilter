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
            filter_: see base.
            num_particles: the number of particles to use for approximating the parameter posteriors.
        """

        super().__init__(filter_)

        self.particles = torch.Size([num_particles])
        self._parameter_shape = torch.Size([num_particles, 1])
        self.filter.set_batch_shape(self.particles)

    def initialize(self) -> SequentialAlgorithmState:
        """
        Initializes the algorithm by returning a :class:`SequentialAlgorithmState`.
        """

        init_state = self.filter.initialize()
        init_weights = torch.zeros(self.particles, device=init_state.get_loglikelihood().device)

        self.context.initialize_parameters(self._parameter_shape)

        return SequentialAlgorithmState(init_weights, self.filter.initialize_with_result(init_state))

    def step(self, y: torch.Tensor, state: SequentialAlgorithmState) -> SequentialAlgorithmState:
        """
        Updates the algorithm and filter state given the latest observation ``y``.

        Args:
            y: the latest observation.
            state: the previous state of the algorithm.

        Returns:
            The updated state of the algorithm.
        """

        raise NotImplementedError()

    def fit(self, y, logging=None, **kwargs) -> SequentialAlgorithmState:
        logging = logging or TQDMWrapper()

        with logging.initialize(self, y.shape[0]):
            state = self.initialize()
            for i, yt in enumerate(y):
                state = self.step(yt, state)
                logging.do_log(i, state)

            return state


class CombinedSequentialParticleAlgorithm(SequentialParticleAlgorithm, ABC):
    """
    Algorithm combining two instances of :class:`SequentialParticleAlgorithm`, where we let one of them target a
    chronological subset of the data, and the other the remaining points.

    One such example is the :class:`pyfilter.inference.sequential.NESSMC2`.
    """

    def __init__(self, filter_, particles, switch: int, first_kw: Dict[str, Any], second_kw: Dict[str, Any]):
        """
        Initializes the :class:`CombinedSequentialParticleAlgorithm` class.

        Args:
            filter_: see base.
            particles: see base.
            switch: the number of observations to have parsed before switching algorithms.
            first_kw: kwargs sent to :meth:`CombinedSequentialParticleAlgorithm.make_first`.
            second_kw: kwargs sent to :meth:`CombinedSequentialParticleAlgorithm.make_second`.
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
            filter_: see ``__init__``.
            particles: see ``__init__``.
            kwargs: corresponds to ``first_kw`` of ``__init__``.

        Returns:
            Instance of algorithm to be used for first part of the data.
        """

        raise NotImplementedError()

    def make_second(self, filter_, particles, **kwargs) -> SequentialParticleAlgorithm:
        """
        See :meth:`CombinedSequentialParticleAlgorithm.make_first` but replace `first` with `second`.
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
