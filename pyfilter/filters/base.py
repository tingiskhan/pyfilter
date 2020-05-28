import copy
from abc import ABC
from ..timeseries import StateSpaceModel
from tqdm import tqdm
import torch
from ..utils import choose
from ..module import Module
from .utils import enforce_tensor, FilterResult
from typing import Tuple, Union


class BaseFilter(Module, ABC):
    def __init__(self, model: StateSpaceModel, save_means=True):
        """
        The basis for filters. Take as input a model and specific attributes.
        :param model: The model
        :param save_means: Whether to record the means, or to ignore
        """

        self._dummy = torch.tensor(0.)

        super().__init__()

        if not isinstance(model, StateSpaceModel):
            raise ValueError(f'`model` must be `{StateSpaceModel.__name__:s}`!')

        self._model = model
        self._n_parallel = None
        self._filter_res = FilterResult()
        self._save_means = save_means

    @property
    def result(self) -> FilterResult:
        """
        Returns the filtering result object.
        """

        return self._filter_res

    @property
    def ssm(self) -> StateSpaceModel:
        """
        Returns the SSM as an object.
        """
        return self._model

    def viewify_params(self, shape):
        """
        Defines views to be used as parameters instead
        :param shape: The shape to use. Please note that
        :type shape: tuple|torch.Size
        :return: Self
        """

        self.ssm.viewify_params(shape)

        return self

    def set_nparallel(self, n: int):
        """
        Sets the number of parallel filters to use
        :param n: The number of parallel filters
        """

        raise NotImplementedError()

    def initialize(self):
        """
        Initializes the filter.
        :return: Self
        """

        return self

    @enforce_tensor
    def filter(self, y: Union[float, torch.Tensor]):
        """
        Performs a filtering the model for the observation `y`.
        :param y: The observation
        :return: Self and log-likelihood
        """

        xm, ll = self._filter(y)

        if self._save_means:
            self._filter_res.append(xm, ll)
        else:
            self._filter_res.append(None, ll)

        return self, ll

    def _filter(self, y: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The actual filtering procedure. Overwrite this.
        :param y: The observation
        :return: Mean of state, log-likelihood
        """

        raise NotImplementedError()

    def longfilter(self, y: Union[torch.Tensor, Tuple[torch.Tensor, ...]], bar=True):
        """
        Filters the entire data set `y`.
        :param y: An array of data. Should be {# observations, # dimensions (minimum of 1)}
        :param bar: Whether to print a progressbar
        :return: Self
        """

        astuple = tuple(y) if not isinstance(y, tuple) else y
        if bar:
            iterator = tqdm(astuple, desc=str(self.__class__.__name__))
        else:
            iterator = astuple

        for yt in iterator:
            self.filter(yt)

        return self

    def copy(self):
        """
        Returns a copy of itself.
        :return: Copy of self
        """

        # TODO: Need to fix the reference to _theta_vals here. If there are parameters we need to redefine them
        return copy.deepcopy(self)

    def predict(self, steps: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts `steps` ahead using the latest available information.
        :param steps: The number of steps forward to predict
        :param kwargs: Any key worded arguments
        """

        raise NotImplementedError()

    def resample(self, inds: torch.Tensor, entire_history: bool = False):
        """
        Resamples the filter, used in cases where we use nested filters.
        :param inds: The indices
        :param entire_history: Whether to resample entire history
        :return: Self
        """
        if entire_history:
            self._filter_res.resample(inds)

        # ===== Resample the parameters of the model ====== #
        self.ssm.p_apply(lambda u: choose(u.values, inds))
        self._resample(inds)

        return self

    def _resample(self, inds: torch.Tensor):
        """
        Implements resampling unique for the filter.
        :param inds: The indices
        :return: Self
        """

        return self

    def reset(self):
        """
        Resets the filter by nullifying the filter specific attributes.
        :return: Self
        """

        self._filter_res = FilterResult()
        self._reset()

        return self

    def _reset(self):
        """
        Any filter specific resets.
        :return: Self
        """

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
        self._filter_res.exchange(filter_._filter_res, inds)
        self._exchange(filter_, inds)

        return self

    def _exchange(self, filter_, inds: torch.Tensor):
        """
        Filter specific exchanges.
        :param filter_: The new filter
        :type filter_: BaseFilter
        :param inds: The indices
        :return: Self
        """

        return self


class BaseKalmanFilter(BaseFilter, ABC):
    def set_nparallel(self, n):
        self._n_parallel = torch.Size([n])

        return self