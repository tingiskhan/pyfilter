import copy
from abc import ABC
from ..timeseries import StateSpaceModel
from tqdm import tqdm
import torch
from ..utils import choose
from torch.nn import Module
from .utils import enforce_tensor, FilterResult
from typing import Tuple, Union, Iterable
from .state import BaseState


class BaseFilter(Module, ABC):
    def __init__(self, model: StateSpaceModel):
        super().__init__()

        if not isinstance(model, StateSpaceModel):
            raise ValueError(f"`model` must be `{StateSpaceModel.__name__:s}`!")

        self._model = model
        self._n_parallel = torch.Size([])

    @property
    def ssm(self) -> StateSpaceModel:
        return self._model

    @property
    def n_parallel(self) -> torch.Size:
        return self._n_parallel

    def viewify_params(self, shape: Union[int, torch.Size]):
        self.ssm.viewify_params(shape)

        return self

    def set_nparallel(self, n: int):
        """
        Sets the number of parallel filters to use
        :param n: The number of parallel filters
        """

        raise NotImplementedError()

    def initialize(self) -> BaseState:
        raise NotImplementedError()

    @enforce_tensor
    def filter(self, y: Union[float, torch.Tensor], state: BaseState) -> BaseState:
        """
        Performs a filtering move given the observation `y`.
        :param y: The observation
        :param state: The previous state
        :return: Self and log-likelihood
        """

        return self._filter(y, state)

    def _filter(self, y: Union[float, torch.Tensor], state: BaseState) -> BaseState:
        raise NotImplementedError()

    def longfilter(
        self,
        y: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        bar=True,
        record_states=False,
        init_state: BaseState = None,
    ) -> FilterResult:
        """
        Filters the entire data set `y`.
        :param y: An array of data. Could either be 1D or 2D.
        :param bar: Whether to print a progressbar
        :param record_states: Whether to record states on a tuple
        :param init_state: The initial state to use
        """

        astuple = tuple(y) if not isinstance(y, tuple) else y
        iterator = tqdm(astuple, desc=str(self.__class__.__name__)) if bar else astuple

        state = init_state or self.initialize()
        result = FilterResult(state)

        for yt in iterator:
            state = self.filter(yt, state)
            result.append(state, not record_states)

        return result

    def copy(self, view_shape=torch.Size([])):
        res = copy.deepcopy(self)
        res.viewify_params(view_shape)
        return res

    def predict(self, state: BaseState, steps: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def resample(self, inds: torch.Tensor):
        """
        Resamples the filter, used in cases where we use nested filters.
        :param inds: The indices
        :return: Self
        """

        self.ssm.p_apply(lambda u: choose(u.values, inds))

        return self

    def exchange(self, filter_, inds: torch.Tensor):
        """
        Exchanges the filters.
        :param filter_: The new filter
        :type filter_: BaseFilter
        :param inds: The indices
        :return: Self
        """

        self._model.exchange(inds, filter_.ssm)

        return self

    def populate_state_dict(self):
        return {"_model": self.ssm.state_dict(), "_n_parallel": self._n_parallel}

    def smooth(self, states: Iterable[BaseState]) -> torch.Tensor:
        raise NotImplementedError()


class BaseKalmanFilter(BaseFilter, ABC):
    def set_nparallel(self, n):
        self._n_parallel = torch.Size([n])

        return self
