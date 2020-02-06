from abc import ABC
from ..filters.base import BaseFilter, enforce_tensor, ParticleFilter
from tqdm import tqdm
import warnings
from ..module import Module, TensorContainer
import torch
from ..utils import normalize


class BaseAlgorithm(Module, ABC):
    def __init__(self, filter_):
        """
        Implements a base class for inference, i.e. inference for inferring parameters.
        :param filter_: The filter
        :type filter_: BaseFilter
        """

        super().__init__()

        self._filter = filter_      # type: BaseFilter
        self._y = tuple()           # type: tuple[torch.Tensor]
        self._iterator = None

    @property
    def filter(self):
        """
        Returns the filter
        :rtype: BaseFilter
        """

        return self._filter

    @filter.setter
    def filter(self, x):
        """
        Sets the filter
        :param x: The new filter
        :type x: BaseFilter
        """

        if not isinstance(x, type(self.filter)):
            raise ValueError('`x` is not {:s}!'.format(type(self.filter)))

        self._filter = x

    def fit(self, y):
        """
        Fits the algorithm to data.
        :param y: The data to fit
        :type y: numpy.ndarray|pandas.DataFrame|torch.Tensor
        :return: Self
        :rtype: BaseAlgorithm
        """

        self._y = y

        raise NotImplementedError()

    def initialize(self):
        """
        Initializes the chosen algorithm.
        :return: Self
        :rtype: BaseAlgorithm
        """

        return self

    def predict(self, steps, *args, **kwargs):
        """
        Predicts `steps` ahead.
        :param steps: The number of steps
        :type steps: int
        :param args: Any arguments
        :param kwargs: Any keyworded arguments
        :rtype: tuple[torch.Tensor]
        """

        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)


class SequentialAlgorithm(BaseAlgorithm, ABC):
    """
    Algorithm for online inference.
    """

    def _update(self, y):
        """
        The function to override by the inherited algorithm.
        :param y: The observation
        :type y: torch.Tensor
        :return: Self
        :rtype: SequentialAlgorithm
        """

        raise NotImplementedError()

    @enforce_tensor
    def update(self, y):
        """
        Performs an update using a single observation `y`.
        :param y: The observation
        :type y: numpy.ndarray|float|torch.Tensor
        :return: Self
        :rtype: SequentialAlgorithm
        """
        self._y += (y.clone(),)
        return self._update(y)

    def fit(self, y, bar=True):
        self._iterator = tqdm(y, desc=str(self))

        for yt in self._iterator if bar else y:
            self.update(yt)

        self._iterator = None

        return self


class SequentialParticleAlgorithm(SequentialAlgorithm, ABC):
    def __init__(self, filter_, particles):
        """
        Implements a base class for sequential particle inference.
        :param particles: The number of particles to use
        :type particles: int
        """
        super().__init__(filter_)
        self._filter.set_nparallel(particles)

        # ===== Weights ===== #
        self._w_rec = None

        # ===== ESS related ===== #
        self._logged_ess = TensorContainer()
        self.particles = particles

    @property
    def particles(self):
        """
        Returns the number of particles.
        :rtype: torch.Tensor
        """

        return self._particles

    @particles.setter
    def particles(self, x):
        """
        Sets the particles.
        """

        self._particles = torch.Size([x])

    def initialize(self):
        """
        Overwrites the initialization.
        :return: Self
        :rtype: NESS
        """

        self.filter.ssm.sample_params(self.particles)
        self._w_rec = torch.zeros(self.particles, device=self.filter._dummy.device)

        shape = torch.Size((*self.particles, 1)) if isinstance(self.filter, ParticleFilter) else self.particles
        self.filter.viewify_params(shape).initialize()

        return self

    @property
    def logged_ess(self):
        """
        Returns the logged ESS.
        :rtype: torch.Tensor
        """

        return torch.tensor(self._logged_ess)

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

    def _fit(self, y):
        """
        The method to override by sub-classes.
        :param y: The data in iterator format
        :type y: iterator
        :return: Self
        :rtype: BatchAlgorithm
        """

        raise NotImplementedError()

    @enforce_tensor
    def fit(self, y):
        self._y = y
        self.initialize()._fit(y)

        return self


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