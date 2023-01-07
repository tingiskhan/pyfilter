from abc import ABC
from typing import Callable, Generic, Sequence, TypeVar, Union

import torch
from stochproc.timeseries import StateSpaceModel
from tqdm import tqdm
from tqdm import tqdm

from .result import FilterResult
from .state import Correction, Prediction

TCorrection = TypeVar("TCorrection", bound=Correction)
TPrediction = TypeVar("TPrediction", bound=Prediction)

BoolOrInt = Union[bool, int]
ModelObject = Union[StateSpaceModel, Callable[["InferenceContext"], StateSpaceModel]]  # noqa: F821


class BaseFilter(Generic[TCorrection, TPrediction]):
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
        Internal initializer for :class:`BaseFilter`.

        Args:
            model (ModelObject): state space model to use for filtering.
            record_states (bool): see :class:`pyfilter.filters.FilterResult`.
            record_moments (bool): see :class:`pyfilter.filters.FilterResult`.
            nan_strategy (str): how to handle ``nan``s in observation data. Can be:
                * "skip" - skips the observation.
                * "impute" - imputes the value using the mean of the predicted distribution. If nested, then uses the
                    median of mean.
            record_intermediary_states (bool): whether to record intermediary states in :meth:`filter` for models where
                `observe_every_step` > 1. Must be `True` whenever you are performing smoothing.
        """

        super().__init__()

        if not (isinstance(model, StateSpaceModel) or callable(model)):
            raise ValueError(f"`model` must be `{StateSpaceModel:s}` or {callable} that returns `{StateSpaceModel:s}!")

        model_is_builder = callable(model) and not isinstance(model, StateSpaceModel)

        if model_is_builder:
            self._model_builder = model
            self._model = None
        else:
            # NB: We use a lambda to avoid having to treat `initialize_model` differently
            self._model_builder = lambda _: model
            self._model = model

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

    def initialize_model(self, context: "InferenceContext"):  # noqa: F821
        r"""
        Initializes the state space model.

        Args:
            context (InferenceContext): context to initialize model with.
        """

        self._model = self._model_builder(context)

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
             batch_shape (torch.Size): batch size.

        Example:
            >>> from pyfilter.filters.particle import SISR
            >>>
            >>> model = ...
            >>>
            >>> sisr = SISR(model, 1_000)
            >>> sisr.set_batch_shape(torch.Size([50]))
            >>>
            >>> state = sisr.initialize()
            >>> state.x.value.shape
            >>> state.x.value.shape
            torch.Size([50, 1000])
        """

        if len(batch_shape) > 1:
            raise NotImplementedError("Currently do not support nested batches!")

        self._batch_shape = batch_shape

    def initialize(self) -> TCorrection:
        """
        Initializes the filter. This is mainly for internal use, consider using
        :meth:`BaseFilter.initialize_with_result` instead.
        """

        raise NotImplementedError()

    def initialize_with_result(self, state: TCorrection = None) -> FilterResult[TCorrection]:
        """
        Initializes the filter using :meth:`BaseFilter.initialize` if ``state`` is ``None``, and wraps the result using
        :class:`~pyfilter.filters.result.FilterResult`.

        Args:
            state (TCorrection): optional parameter, if ``None`` calls :meth:`BaseFilter..initialize` otherwise uses ``state``.
        """

        return FilterResult(state or self.initialize(), self.record_states, self.record_moments)

    def batch_filter(self, y: Sequence[torch.Tensor], bar=True, init_state: TCorrection = None) -> FilterResult[TCorrection]:
        """
        Batch version of :meth:`BaseFilter.filter` where entire data set is parsed.

        Args:
            y (torch.Tensor): data to filter..
            bar (bool): whether to display a ``tqdm`` progress bar.
            init_state (bool): optional parameter for whether to pass an initial state.
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

    def predict(self, state: TCorrection) -> TPrediction:
        """
        Corresponds to the predict step of the filter.

        Args:
            state (TCorrection): previous state of the filter algorithm.
        """

        raise NotImplementedError()

    def correct(self, y: torch.Tensor, prediction: TPrediction) -> TCorrection:
        """
        Corresponds to the correct step of the filter.

        Args:
            y (torch.Tensor): observation to use when correcting for.
            prediction (Prediction): predicted state of the filter.
        """

        raise NotImplementedError()

    def filter(self, y: torch.Tensor, correction: TCorrection, result: FilterResult = None) -> TCorrection:
        """
        Performs one filter move given observation ``y`` and previous state of the filter.

        Args:
            y (torch.Tensor): see :meth:`correct`.
            correction (TCorrection): see :meth:`predict`.
            result (FilterResult): optional parameter specifying the result on which to append the resulting states.

        Returns:
            TCorrection: corrected state.
        """

        prediction = self.predict(correction)

        result_is_none = result is None
        while prediction.get_timeseries_state().time_index % self._model.observe_every_step != 0:
            correction = prediction.create_state_from_prediction(self._model)

            if not result_is_none and self._record_intermediary:
                result.append(correction)

            prediction = self.predict(correction)

        nan_mask = torch.isnan(y)
        if nan_mask.all():
            correction = prediction.create_state_from_prediction(self._model)
        else:
            correction = self.correct(y, prediction)

        if not result_is_none:
            result.append(correction)

        return correction

    def smooth(self, states: Sequence[TCorrection], method: str) -> torch.Tensor:
        """
        Smooths the estimated trajectory by sampling from :math:`p(x_{1:t} | y_{1:t})`.

        Args:
            states (Sequence[TCorrection]): states obtained by performing filtering.
            method (str): method to use for smoothing.
        """

        raise NotImplementedError()
