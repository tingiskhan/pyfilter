from abc import ABC
from tqdm import tqdm
import torch
from typing import Tuple, Sequence, TypeVar, Union, Callable
from stochproc.timeseries import StateSpaceModel

from .result import FilterResult
from .state import FilterState, PredictionState


TState = TypeVar("TState", bound=FilterState)
BoolOrInt = Union[bool, int]
ModelObject = Union[StateSpaceModel, Callable[["ParameterContext"], StateSpaceModel]]


class BaseFilter(ABC):
    """
    Abstract base class for filters.
    """

    def __init__(
        self,
        model: ModelObject,
        record_states: BoolOrInt = False,
        record_moments: BoolOrInt = True,
        nan_strategy: str = "skip",
        record_intermediary_states: bool = False,
    ):
        """
        Initializes the :class:`BaseFilter` class.

        Args:
            model: the state space model to use for filtering.
            record_states: see :class:`pyfilter.filters.FilterResult`.
            record_moments: see :class:`pyfilter.filters.FilterResult`.
            nan_strategy: how to handle ``nan``s in observation data. Can be:
                * "skip" - skips the observation.
                * "impute" - imputes the value using the mean of the predicted distribution. If nested, then uses the
                    median of mean.
            record_intermediary_states: whether to record intermediary states in :meth:`filter` for models where
                `observe_every_step` > 1. Must be `True` whenever you are performing smoothing.
        """

        from ..inference.context import ParameterContext

        super().__init__()

        if not (isinstance(model, StateSpaceModel) or callable(model)):
            raise ValueError(f"`model` must be `{StateSpaceModel:s}` or {callable} that returns `{StateSpaceModel:s}!")

        is_function = callable(model) and not isinstance(model, StateSpaceModel)

        self._model_builder = model if is_function else None

        self._model = model if not is_function else model(ParameterContext.get_context())
        self._batch_shape = torch.Size([])

        self.record_states = record_states
        self.record_moments = record_moments

        if nan_strategy not in ["skip", "impute"]:
            raise NotImplementedError(f"Currently cannot handle strategy '{nan_strategy}'!")

        self._nan_strategy = nan_strategy
        self._record_intermediary = record_intermediary_states

    @property
    def ssm(self) -> StateSpaceModel:
        return self._model

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the number of parallel filters.
        """

        return self._batch_shape

    def set_batch_shape(self, batch_shape: torch.Size):
        """
        Sets the number of parallel filters to use by utilizing broadcasting. Useful when running sequential particle
        algorithms or multiple parallel chains of MCMC, as this avoids the linear cost of iterating over multiple filter
        objects.

        Args:
             batch_shape: batch size.

        Example:
            >>> from pyfilter.filters.particle import SISR
            >>>
            >>> model = ...
            >>>
            >>> sisr = SISR(model, 1_000)
            >>> sisr.set_batch_shape(torch.Size([50]))
            >>>
            >>> state = sisr.initialize()
            >>> state.x.values.shape
            torch.Size([50, 1000])
        """

        if len(batch_shape) > 1:
            raise NotImplementedError("Currently do not support nested batches!")

        self._batch_shape = batch_shape

    def initialize(self) -> TState:
        """
        Initializes the filter. This is mainly for internal use, consider using
        :meth:`BaseFilter.initialize_with_result` instead.
        """

        raise NotImplementedError()

    def initialize_with_result(self, state: TState = None) -> FilterResult[TState]:
        """
        Initializes the filter using :meth:`BaseFilter.initialize` if ``state`` is ``None``, and wraps the result using
        :class:`~pyfilter.filters.result.FilterResult`.

        Args:
            state: optional parameter, if ``None`` calls :meth:`BaseFilter..initialize` otherwise uses ``state``.
        """

        return FilterResult(state or self.initialize(), self.record_states, self.record_moments)

    def batch_filter(self, y: Sequence[torch.Tensor], bar=True, init_state: TState = None) -> FilterResult[TState]:
        """
        Batch version of :meth:`BaseFilter.filter` where entire data set is parsed.

        Args:
            y: data set to filter.
            bar: whether to display a ``tqdm`` progress bar.
            init_state: optional parameter for whether to pass an initial state.
        """

        iter_bar = y if not bar else tqdm(desc=str(self.__class__.__name__), total=len(y))

        try:
            state = init_state or self.initialize()
            result = self.initialize_with_result(state)

            for yt in y:
                state = self.filter(yt, state, result=result)

                if bar:
                    iter_bar.update(1)

            return result
        except Exception as e:
            raise e
        finally:
            if bar:
                iter_bar.close()

    def copy(self) -> "BaseFilter":
        """
        Creates a copy of the filter object.
        """

        raise NotImplementedError()

    def predict(self, state: TState) -> PredictionState:
        """
        Corresponds to the predict step of the given filter.

        Args:
            state: the previous state of the algorithm.
        """

        raise NotImplementedError()

    def correct(self, y: torch.Tensor, state: TState, prediction: PredictionState) -> TState:
        """
        Corresponds to the correct step of the given filter.

        Args:
            y: the observation.
            state: the previous state of the algorithm.
            prediction: the predicted state.
        """

        raise NotImplementedError()

    def filter(self, y: torch.Tensor, state: TState, result: FilterResult = None) -> TState:
        """
        Performs one filter move given observation ``y`` and previous state of the filter.

        Args:
            y: the observation for which to filter.
            state: the previous state of the filter.
            result: optional parameter specifying the result on which to append the resulting states.

        Returns:
            Updated state.
        """

        prediction = self.predict(state)

        result_is_none = result is None
        # TODO: Would be neat to record the intermediary results to the result object such that we may perform smoothing
        while prediction.get_previous_state().time_index % self._model.observe_every_step != 0:
            state = prediction.create_state_from_prediction(self._model)

            if not result_is_none and self._record_intermediary:
                result.append(state)

            prediction = self.predict(state)

        nan_mask = torch.isnan(y)
        if nan_mask.all():
            state = prediction.create_state_from_prediction(self._model)
        else:
            state = self.correct(y, state, prediction)

        if not result_is_none:
            result.append(state)

        return state

    def smooth(self, states: Sequence[FilterState]) -> torch.Tensor:
        """
        Smooths the estimated trajectory by sampling from :math:`p(x_{1:t} | y_{1:t})`.

        Args:
            states: the filtered states.
        """

        raise NotImplementedError()
