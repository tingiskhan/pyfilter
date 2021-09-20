import copy
from abc import ABC
from tqdm import tqdm
import torch
from torch.nn import Module
from typing import Tuple, Iterable, TypeVar, Union, List, Callable
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
            record_states: Optional parameter for whether to record all, or some of the
                ``pyfilter.filters.state.BaseFilterState`` objects. Can be either a ``bool``  or an ``int``, if ``int``
                the ``pyfilter.filters.result.FilterResult`` object will retain ``record_states`` number of states. If
                ``True`` will retain *all* states, and only the latest if ``False``. Do note that recording all states
                will be very memory intensive for particle filters.
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

    def set_nparallel(self, num_filters: int):
        """
        Sets the number of parallel filters to use by utilizing broadcasting. Useful when running sequential particle
        algorith or multiple parallel chains of MCMC, as this avoids the linear cost of iterating over multiple filter
        objects.

        Args:
             num_filters: The number of filters to run in parallel.
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
            state: Optional parameter, if ``None`` calls ``.initialize()`` otherwise uses ``state``
        """

        res = FilterResult(state or self.initialize(), self.record_states)

        for callback in self._pre_append_callbacks:
            res.register_forward_pre_hook(callback)

        return res

    def filter(self, y: torch.Tensor, state: TState) -> TState:
        """
        Performs a filtering move given observation `y` and previous state of the filter. Wraps the ``__call__`` method
        of `torch.nn.Module``.

        Args:
            y: The next observation
        """

        return self.__call__(y, state)

    def longfilter(self, y: Iterable[torch.Tensor], bar=True, init_state: TState = None) -> FilterResult[TState]:
        """
        Filters the entire data set `y`.

        :param y: Data, expected shape: (# time steps, [# dimensions])
        :param bar: Whether to print a progress bar
        :param init_state: The initial state to use
        """

        iter_bar: tqdm = None
        if bar:
            iter_bar = tqdm(desc=str(self.__class__.__name__), total=y.shape[0])

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
        return copy.deepcopy(self)

    def predict(self, state: TState, steps: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def predict_correct(self, y: torch.Tensor, state: TState) -> TState:
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        return self.predict_correct(*args, **kwargs)

    def resample(self, indices: torch.Tensor) -> "BaseFilter":
        """
        Resamples the filter, used in when we have nested filters.
        """

        for m in [self.ssm.hidden, self.ssm.observable]:
            for p in m.parameters():
                p[:] = choose(p, indices)

        return self

    def exchange(self, filter_: "BaseFilter", indices: torch.Tensor):
        """
        Exchanges the filters, used when we have have nested filters.
        """

        self._model.exchange(indices, filter_.ssm)

        return self

    def smooth(self, states: Tuple[BaseFilterState]) -> torch.Tensor:
        raise NotImplementedError()
