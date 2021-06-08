import copy
from abc import ABC
from tqdm import tqdm
import torch
from torch.nn import Module
from typing import Tuple, Iterable, TypeVar, Optional
from ..timeseries import StateSpaceModel
from ..utils import choose
from .result import FilterResult
from .state import BaseState


TState = TypeVar("TState", bound=BaseState)


class BaseFilter(Module, ABC):
    """
    Base class for filters.
    """

    def __init__(self, model: StateSpaceModel):
        super().__init__()

        if not isinstance(model, StateSpaceModel):
            raise ValueError(f"`model` must be `{StateSpaceModel.__name__:s}`!")

        self._model = model
        self.register_buffer("_n_parallel", torch.tensor(0, dtype=torch.int))
        self._n_parallel = None

    @property
    def ssm(self) -> StateSpaceModel:
        return self._model

    @property
    def n_parallel(self) -> torch.Size:
        if self._n_parallel is None or self._n_parallel == 0:
            return torch.Size([])

        return torch.Size([self._n_parallel])

    def set_nparallel(self, num_filters: int):
        """
        Sets the number of parallel filters to use
        """

        raise NotImplementedError()

    def initialize(self) -> TState:
        raise NotImplementedError()

    def filter(self, y: torch.Tensor, state: TState) -> TState:
        """
        Performs a filtering move given observation `y` and previous state of the filter.
        """

        return self.predict_correct(y, state)

    def longfilter(
        self, y: Iterable[torch.Tensor], bar=True, record_states=False, init_state: BaseState = None,
    ) -> FilterResult:
        """
        Filters the entire data set `y`.

        :param y: Data, expected shape: (# time steps, [# dimensions])
        :param bar: Whether to print a progress bar
        :param record_states: Whether to record states. E.g. required when smoothing
        :param init_state: The initial state to use
        """

        iter_bar: tqdm = None
        if bar:
            iter_bar = tqdm(desc=str(self.__class__.__name__), total=y.shape[0])

        try:
            state = init_state or self.initialize()
            result = FilterResult(state)

            for yt in y:
                state = self.filter(yt, state)

                if bar:
                    iter_bar.update(1)

                result.append(state, not record_states)

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

    def predict_correct(self, y: Optional[torch.Tensor], state: TState) -> TState:
        raise NotImplementedError()

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

    def smooth(self, states: Iterable[BaseState]) -> torch.Tensor:
        raise NotImplementedError()
