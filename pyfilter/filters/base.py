import copy
from abc import ABC
from tqdm import tqdm
import torch
from torch.nn import Module
from typing import Tuple, Sequence, TypeVar, Union, List, Callable
from ..timeseries import StateSpaceModel
from ..utils import choose
from .result import FilterResult
from .state import BaseFilterState


TState = TypeVar("TState", bound=BaseFilterState)


class BaseFilter(Module, ABC):
    """
    Abstract base class for filters.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        record_states: Union[bool, int] = False,
        pre_append_callbacks: List[Callable[[TState], None]] = None,
    ):
        """
        Initializes the filter object.

        Args:
            model: The state space model to use for filtering.
            record_states: See ``pyfilter.timeseries.result.record_states``.
            pre_append_callbacks: Any callbacks that will be executed by ``pyfilter.filters.result.FilterResult`` prior
                to appending the new state.
        """

        super().__init__()

        if not isinstance(model, StateSpaceModel):
            raise ValueError(f"`model` must be `{StateSpaceModel.__name__:s}`!")

        self._model = model
        self.register_buffer("_n_parallel", torch.tensor(0, dtype=torch.int))
        self._n_parallel = None
        self.record_states = record_states

        self._pre_append_callbacks = pre_append_callbacks or list()

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
        Initializes the filter.
        """

        raise NotImplementedError()

    def initialize_with_result(self, state: TState = None) -> FilterResult[TState]:
        """
        Initializes the filter using ``.initialize()`` if ``state`` is ``None``, and wraps the result using
        ``pyfilter.filters.result.FilterResult``. Also registers the callbacks on the ``FilterResult`` object.

        Args:
            state: Optional parameter, if ``None`` calls ``.initialize()`` otherwise uses ``state``.
        """

        res = FilterResult(state or self.initialize(), self.record_states)

        for callback in self._pre_append_callbacks:
            res.register_forward_pre_hook(callback)

        return res

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
            init_state: Optional parameter for whether to pass an initial state
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

    def predict(self, state: TState, steps: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the previous ``state``, predict ``steps`` steps into the future.

        Args:
              state: Previous state.
              steps: The number of steps to predict.
              args: Any filter specific arguments.
              kwargs: Any filter specific kwargs.

        Returns:
            Returns a tuple consisting of (predicted x, predicted y).
        """

        raise NotImplementedError()

    def forward(self, y: torch.Tensor, state: TState) -> TState:
        """
        Method to be overridden by derived filters.

        Args:
            y: See ``self.filter(...)``.
            state: See ``self.filter(...)``.
        """

        raise NotImplementedError()

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

    def smooth(self, states: Sequence[BaseFilterState]) -> torch.Tensor:
        """
        Smooths the estimated trajectory by sampling from :math:`p(x_{1:t} | y_{1:t})`.

        Args:
            states: The filtered states.
        """

        raise NotImplementedError()
