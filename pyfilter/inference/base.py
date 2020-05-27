from abc import ABC
from ..filters.base import BaseFilter, enforce_tensor, ParticleFilter
from tqdm import tqdm
import warnings
from ..module import Module, TensorContainer
import torch
from ..utils import normalize
from typing import Tuple


class BaseAlgorithm(Module, ABC):
    def __init__(self):
        """
        Implements a base class for inference.
        """

        super().__init__()

        self._iterator = None

    @enforce_tensor
    def fit(self, y: torch.Tensor):
        """
        Fits the algorithm to data.
        :param y: The data to fit
        :return: Self
        """

        self._fit(y)

        return self

    def _fit(self, y: torch.Tensor):
        """
        Method to be overridden by user.
        """

        raise NotImplementedError()

    def initialize(self):
        """
        Initializes the chosen algorithm.
        :return: Self
        """

        return self

    def predict(self, steps: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts `steps` ahead.
        :param steps: The number of steps
        :param args: Any arguments
        :param kwargs: Any keyworded arguments
        """

        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)


class BaseFilterAlgorithm(BaseAlgorithm, ABC):
    def __init__(self, filter_: BaseFilter):
        """
        Base class for algorithms utilizing filters for inference.
        :param filter_: The filter
        :type filter_: BaseFilter
        """

        super().__init__()

        self._filter = filter_

    @property
    def filter(self) -> BaseFilter:
        """
        Returns the filter
        """

        return self._filter

    @filter.setter
    def filter(self, x: BaseFilter):
        """
        Sets the filter
        :param x: The new filter
        """

        if not isinstance(x, type(self.filter)):
            raise ValueError('`x` is not {:s}!'.format(type(self.filter)))

        self._filter = x


class SequentialAlgorithm(BaseFilterAlgorithm, ABC):
    """
    Algorithm for sequential inference.
    """

    def _update(self, y: torch.Tensor) -> BaseFilterAlgorithm:
        """
        The function to override by the inherited algorithm.
        :param y: The observation
        :return: Self
        """

        raise NotImplementedError()

    @enforce_tensor
    def update(self, y: torch.Tensor) -> BaseFilterAlgorithm:
        """
        Performs an update using a single observation `y`.
        :param y: The observation
        :return: Self
        """

        return self._update(y)

    def _fit(self, y, bar=True):
        iterator = y
        if bar:
            self._iterator = iterator = tqdm(y, desc=str(self))

        for yt in iterator:
            self.update(yt)

        self._iterator = None

        return self


class SequentialParticleAlgorithm(SequentialAlgorithm, ABC):
    def __init__(self, filter_, particles: int):
        """
        Implements a base class for sequential particle inference.
        :param particles: The number of particles to use
        """

        super().__init__(filter_)

        # ===== Weights ===== #
        self._w_rec = None

        # ===== ESS related ===== #
        self._logged_ess = TensorContainer()
        self.particles = particles

    @property
    def particles(self) -> torch.Size:
        """
        Returns the number of particles.
        """

        return self._particles

    @particles.setter
    def particles(self, x: int):
        """
        Sets the particles.
        """

        self._particles = torch.Size([x])

    def initialize(self) -> BaseFilterAlgorithm:
        """
        Overwrites the initialization.
        :return: Self
        """

        self._filter.set_nparallel(*self.particles)

        self.filter.ssm.sample_params(self.particles)
        self._w_rec = torch.zeros(self.particles, device=self.filter._dummy.device)

        shape = torch.Size((*self.particles, 1)) if isinstance(self.filter, ParticleFilter) else self.particles
        self.filter.viewify_params(shape).initialize()

        return self

    @property
    def logged_ess(self) -> torch.Tensor:
        """
        Returns the logged ESS.
        """

        return torch.stack(self._logged_ess.tensors)

    def predict(self, steps, aggregate=True, **kwargs):
        px, py = self.filter.predict(steps, aggregate=aggregate, **kwargs)

        if not aggregate:
            return px, py

        w = normalize(self._w_rec)
        wsqd = w.unsqueeze(-1)

        xm = (px * (wsqd if self.filter.ssm.hidden_ndim > 1 else w)).sum(1)
        ym = (py * (wsqd if self.filter.ssm.obs_ndim > 1 else w)).sum(1)

        return xm, ym


class BatchAlgorithm(BaseAlgorithm, ABC):
    """
    Algorithm for batch inference.
    """


class BatchFilterAlgorithm(BaseFilterAlgorithm, ABC):
    def __init__(self, filter_):
        """
        Implements a class of inference algorithms using filters for inference.
        """

        super().__init__(filter_)


def experimental(func):
    def wrapper(obj, *args, **kwargs):
        warnings.warn('{:s} is an experimental algorithm, use at own risk'.format(str(obj)))

        return func(obj, *args, **kwargs)

    return wrapper


def preliminary(func):
    def wrapper(obj, *args, **kwargs):
        warnings.warn('{:s} is only a preliminary version algorithm, use at own risk'.format(str(obj)))

        return func(obj, *args, **kwargs)

    return wrapper