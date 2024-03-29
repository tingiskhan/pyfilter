from abc import ABC
from typing import Any, Callable, Dict, List, TypeVar

import torch

from ..base import BaseAlgorithm
from ..logging import TQDMWrapper
from .state import SequentialAlgorithmState

T = TypeVar("T", bound=SequentialAlgorithmState)
Callback = Callable[["SequentialParticleAlgorithm", torch.Tensor, T], None]


class SequentialParticleAlgorithm(BaseAlgorithm, ABC):
    """
    Abstract base class for sequential algorithms using filters in order to approximate the log likelihood.
    """

    def __init__(self, filter_, num_particles: int, context=None):
        """
        Internal initializer for :class:`SequentialParticleAlgorithm`.

        Args:
            filter_ (BaseFilter): see :class:`BaseAlgorithm`.
            num_particles (int): umber of particles to use for approximating the parameter posteriors.
            context (InferenceContext, optional): inference context. Defaults to None.
        """

        super().__init__(filter_, context=context)

        self.particles = torch.Size([num_particles])

        self.filter.set_batch_shape(self.particles)
        self.context.set_batch_shape(self.particles)

        self._callbacks: List[Callback] = list()

    def register_callback(self, callback: Callback):
        """
        Registers a callback that is called directly after :meth:`step`.

        Args:
            callback (Callback): callback to register.
        """

        if (callback in self._callbacks) or (callback is None):
            return

        self._callbacks.append(callback)

    def initialize(self) -> SequentialAlgorithmState:
        """
        Internal initializer for algorithm by returning a :class:`SequentialAlgorithmState`.
        """

        self.filter.initialize_model(self.context)
        self.context.initialize_parameters()

        init_state = self.filter.initialize()
        init_weights = torch.zeros(self.particles, device=init_state.get_loglikelihood().device)

        return SequentialAlgorithmState(init_weights, self.filter.initialize_with_result(init_state))

    def step(self, y: torch.Tensor, state: SequentialAlgorithmState) -> SequentialAlgorithmState:
        """
        Updates the algorithm and filter state given the latest observation ``y``.

        Args:
            y (torch.Tensor): latest observation.
            state (SequentialAlgorithmState): previous state of the algorithm.
        """

        result = self._step(y, state)

        for cb in self._callbacks:
            cb(self, y, state)

        result.bump_iteration()

        return result

    def _step(self, y: torch.Tensor, state: SequentialAlgorithmState) -> SequentialAlgorithmState:
        """
        Defines how to update ``state`` given the latest observation ``y``, should be overridden by derived classes.

        Args:
            y (torch.Tensor): latest observation.
            state (SequentialAlgorithmState): previous state of the algorithm.
        """

        raise NotImplementedError()

    def fit(self, y, logging=None) -> SequentialAlgorithmState:
        logging = logging or TQDMWrapper()

        with logging.initialize(self, y.shape[0]):
            state = self.initialize()
            for yt in y:
                state = self.step(yt, state)
                logging.do_log(state.current_iteration, state)

            return state


class CombinedSequentialParticleAlgorithm(SequentialParticleAlgorithm):
    """
    Algorithm combining two instances of :class:`SequentialParticleAlgorithm`, where we let one of them target a
    chronological subset of the data, and the other the remaining points.

    One such example is the :class:`pyfilter.inference.sequential.NESSMC2`.
    """

    def __init__(
        self, filter_, num_particles, switch: int, first_kw: Dict[str, Any], second_kw: Dict[str, Any], context=None
    ):
        """
        Internal initializer for :class:`CombinedSequentialParticleAlgorithm`.

        Args:
            filter_ (BaseFilter): see :class:`SequentialParticleAlgorithm`.
            particles (int): see :class:`SequentialParticleAlgorithm`.
            switch (int): number of observations to have parsed before switching algorithms.
            first_kw (Dict[str, Any]): kwargs sent to :meth:`CombinedSequentialParticleAlgorithm.make_first`.
            second_kw (Dict[str, Any]): kwargs sent to :meth:`CombinedSequentialParticleAlgorithm.make_second`.
            context (InferenceContext, optional): see :class:`SequentialParticleAlgorithm`.. Defaults to None.
        """

        super().__init__(filter_=filter_, num_particles=num_particles, context=context)

        self._first = self.make_first(filter_, context, num_particles, **(first_kw or dict()))
        self._second = self.make_second(filter_, context, num_particles, **(second_kw or dict()))

        self._when_to_switch = switch
        self._is_switched = False

    def make_first(self, filter_, context, particles, **kwargs) -> SequentialParticleAlgorithm:
        """
        Creates the algorithm to be used for the first part of the data.

        Args:
            filter_ (BaseFilter): see :meth:`__init__`.
            context (InferenceContext): see :meth:`__init__`.
            particles (int): see :meth:`__init__`.
        """

        raise NotImplementedError()

    def make_second(self, filter_, context, particles, **kwargs) -> SequentialParticleAlgorithm:
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

    def _step(self, y: torch.Tensor, state):
        if not self._is_switched:
            if state.current_iteration <= self._when_to_switch:
                return self._first._step(y, state)

            self._is_switched = True
            state = self.do_on_switch(self._first, self._second, state)

        return self._second._step(y, state)
