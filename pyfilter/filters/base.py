import copy
from abc import ABC
from tqdm import tqdm
import torch
from torch.nn import Module
from typing import Tuple, Sequence, TypeVar, List, Callable
from torch.distributions import Distribution
from .utils import select_mean_of_dist
from ..timeseries import StateSpaceModel
from ..utils import choose
from .result import FilterResult
from .state import FilterState, PredictionState
from ..container import BoolOrInt


TState = TypeVar("TState", bound=FilterState)


class BaseFilter(Module, ABC):
    """
    Abstract base class for filters.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        record_states: BoolOrInt = False,
        record_moments: BoolOrInt = True,
        nan_strategy: str = "skip",
    ):
        """
        Initializes the ``BaseFilter`` class.

        Args:
            model: The state space model to use for filtering.
            record_states: See ``pyfilter.filters.FilterResult.record_states``.
            record_moments: See ``pyfilter.filters.FilterResult.record_moments``.
            nan_strategy: How to handle ``nan``s in observation data. Can be:
                * "skip" - skips the observation.
                * "impute" - imputes the value using the mean of the predicted distribution. If nested, then uses the
                    median of mean.
        """

        super().__init__()

        if not isinstance(model, StateSpaceModel):
            raise ValueError(f"`model` must be `{StateSpaceModel.__name__:s}`!")

        self._model = model
        self.register_buffer("_n_parallel", torch.tensor(0, dtype=torch.int))
        self._n_parallel = None

        self.record_states = record_states
        self.record_moments = record_moments

        if nan_strategy not in ["skip", "impute"]:
            raise NotImplementedError(f"Currently cannot handle strategy '{nan_strategy}'!")

        self._nan_strategy = nan_strategy

    @property
    def ssm(self) -> StateSpaceModel:
        return self._model

    @property
    def n_parallel(self) -> torch.Size:
        """
        Returns the number of parallel filters.
        """

        if self._n_parallel is None or self._n_parallel == 0:
            return torch.Size([])

        return torch.Size([self._n_parallel])

    def set_num_parallel(self, num_filters: int):
        """
        Sets the number of parallel filters to use by utilizing broadcasting. Useful when running sequential particle
        algorithms or multiple parallel chains of MCMC, as this avoids the linear cost of iterating over multiple filter
        objects.

        Args:
             num_filters: The number of filters to run in parallel.

        Example:
            >>> from pyfilter.filters.particle import SISR
            >>>
            >>> model = ...
            >>>
            >>> sisr = SISR(model, 1000)
            >>> sisr.set_num_parallel(50)
            >>>
            >>> state = sisr.initialize()
            >>> state.x.values.shape
            torch.Size([50, 1000])
        """

        raise NotImplementedError()

    def initialize(self) -> TState:
        """
        Initializes the filter. This is mainly for internal use, consider using `.`initialize_with_result(...)``
        instead.
        """

        raise NotImplementedError()

    def initialize_with_result(self, state: TState = None) -> FilterResult[TState]:
        """
        Initializes the filter using ``.initialize()`` if ``state`` is ``None``, and wraps the result using
        ``pyfilter.filters.result.FilterResult``. Also registers the callbacks on the ``FilterResult`` object.

        Args:
            state: Optional parameter, if ``None`` calls ``.initialize()`` otherwise uses ``state``.
        """

        return FilterResult(state or self.initialize(), self.record_states, self.record_moments)

    def filter(self, y: torch.Tensor, state: TState) -> TState:
        """
        Performs one filter move given observation ``y`` and previous state of the filter. Wraps the ``__call__``
        method of `torch.nn.Module``.

        Args:
            y: The observation for which to filter.
            state: The previous state of the filter.

        Returns:
            New and updated state.
        """

        return self.__call__(y, state)

    def longfilter(self, y: Sequence[torch.Tensor], bar=True, init_state: TState = None) -> FilterResult[TState]:
        """
        Batch version of ``.filter(...)`` where entire data set is parsed.

        Args:
            y: Data set to filter.
            bar: Whether to display a ``tqdm`` progress bar.
            init_state: Optional parameter for whether to pass an initial state.
        """

        iter_bar = y if not bar else tqdm(desc=str(self.__class__.__name__), total=len(y))

        try:
            state = init_state or self.initialize()
            result = self.initialize_with_result(state)

            for yt in y:
                state = self.filter(yt, state)

                if bar:
                    iter_bar.update(1)

                result.append(state)

            return result
        except Exception as e:
            raise e
        finally:
            if bar:
                iter_bar.close()

    def copy(self) -> "BaseFilter":
        """
        Creates a deep copy of the filter object.
        """

        return copy.deepcopy(self)

    def predict_path(self, state: TState, steps: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the previous ``state``, predict ``steps`` steps into the future.

        Args:
              state: Previous state.
              steps: The number of steps to predict.
              args: Any filter specific arguments.
              kwargs: Any filter specific kwargs.

        Returns:
            Returns a tuple consisting of ``(predicted x, predicted y)``, where ``x`` and ``y`` are of size
            ``(steps, [additional shapes])``.
        """

        raise NotImplementedError()

    def _get_observation_dist_from_prediction(self, prediction: PredictionState) -> Distribution:
        """
        Method for generating an observation distribution from the predicted latent distribution.

        Args:
            prediction: The prediction to use for creating the distribution.
        """

        raise NotImplementedError()

    def predict(self, state: TState) -> PredictionState:
        """
        Corresponds to the predict step of the given filter.

        Args:
            state: The previous state of the algorithm.
        """

        raise NotImplementedError()

    def correct(self, y: torch.Tensor, state: TState, prediction: PredictionState) -> TState:
        """
        Corresponds to the correct step of the given filter.

        Args:
            y: The observation.
            state: The previous state of the algorithm.
            prediction: The predicted state.
        """

        raise NotImplementedError()

    def forward(self, y: torch.Tensor, state: TState) -> TState:
        """
        Method to be overridden by derived filters.

        Args:
            y: See ``self.filter(...)``.
            state: See ``self.filter(...)``.
        """

        prediction = self.predict(state)

        nan_mask = torch.isnan(y)
        if nan_mask.any():
            # TODO: Perhaps handle switching in __init__ instead?
            if self._nan_strategy == "skip":
                return prediction.create_state_from_prediction()
            elif self._nan_strategy == "impute":
                dist = self._get_observation_dist_from_prediction(prediction)

                # NB: Might be better to reshape `y` to the number of parallel filters instead of using global mean?
                mean = select_mean_of_dist(dist)
                if len(dist.batch_shape) > 0:
                    for d in reversed(range(0, len(dist.batch_shape))):
                        mean = mean.median(dim=d)[0]

                y = y.clone()
                y[nan_mask] = mean[nan_mask]

        return self.correct(y, state, prediction)

    def resample(self, indices: torch.Tensor) -> "BaseFilter":
        """
        Resamples the parameters of the ``.ssm`` attribute, used e.g. when running parallel filters.

        Args:
             indices: The indices to select.
        """

        if self.n_parallel.numel() == 0:
            raise Exception("No parallel filters, cannot resample!")

        for m in [self.ssm.hidden, self.ssm.observable]:
            for p in m.parameters():
                p[:] = choose(p, indices)

        return self

    def exchange(self, filter_: "BaseFilter", indices: torch.Tensor):
        """
        Exchanges the parameters of ``.ssm`` with the parameters of ``filter_.ssm`` at the locations specified by
        ``indices``.

        Args:
            filter_: The filter to exchange parameters with.
            indices: Mask specifying which parallel filters to exchange.
        """

        if self.n_parallel.numel() == 0:
            raise Exception("No parallel filters, cannot resample!")

        self._model.exchange(indices, filter_.ssm)

        return self

    def smooth(self, states: Sequence[FilterState]) -> torch.Tensor:
        """
        Smooths the estimated trajectory by sampling from :math:`p(x_{1:t} | y_{1:t})`.

        Args:
            states: The filtered states.
        """

        raise NotImplementedError()
