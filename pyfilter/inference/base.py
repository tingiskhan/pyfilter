from abc import ABC
from ..filters import BaseFilter, ParticleFilter, utils as u
from ..module import Module, TensorContainer
import torch
from ..utils import normalize
from typing import Tuple
from ..logging import LoggingWrapper, DefaultLogger, TqdmWrapper


class BaseAlgorithm(Module, ABC):
    def __init__(self):
        """
        Implements a base class for inference.
        """

        super().__init__()

    @u.enforce_tensor
    def fit(self, y: torch.Tensor, logging: LoggingWrapper = None, **kwargs):
        """
        Fits the algorithm to data.
        :param y: The data to fit
        :param logging: The logging wrapper
        :return: Self
        """

        self._fit(y, logging_wrapper=logging or TqdmWrapper(), **kwargs)

        return self

    def _fit(self, y: torch.Tensor, logging_wrapper: LoggingWrapper, **kwargs):
        """
        Method to be overridden by user.
        """

        raise NotImplementedError()

    def initialize(self, *args, **kwargs):
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

    @u.enforce_tensor
    def update(self, y: torch.Tensor) -> BaseFilterAlgorithm:
        """
        Performs an update using a single observation `y`.
        :param y: The observation
        :return: Self
        """

        return self._update(y)

    def _fit(self, y, logging_wrapper=None, **kwargs):
        logging_wrapper.set_num_iter(y.shape[0])

        for i, yt in enumerate(y):
            self.update(yt)
            logging_wrapper.do_log(i, self, y)

        return self


class SequentialParticleAlgorithm(SequentialAlgorithm, ABC):
    def __init__(self, filter_, particles: int):
        """
        Implements a base class for sequential particle inference.
        :param particles: The number of particles to use
        """

        super().__init__(filter_)

        # ===== Weights ===== #
        self._w_rec = None  # type: torch.Tensor

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
        self.filter.set_nparallel(x)

    def viewify_params(self):
        shape = torch.Size((*self.particles, 1)) if isinstance(self.filter, ParticleFilter) else self.particles
        self.filter.viewify_params(shape).initialize()

        return self

    def initialize(self) -> BaseFilterAlgorithm:
        """
        Overwrites the initialization.
        :return: Self
        """

        self.filter.set_nparallel(*self.particles)  # Need this line when reinitializing, not optimal...
        self.filter.ssm.sample_params(self.particles)
        self._w_rec = torch.zeros(self.particles, device=self.filter._dummy.device)

        self.viewify_params()

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
    def __init__(self, max_iter: int):
        """
        Algorithm for batch inference.
        """
        super(BatchAlgorithm, self).__init__()
        self._max_iter = int(max_iter)

    def is_converged(self, old_loss, new_loss):
        raise NotImplementedError()

    def _fit(self, y: torch.Tensor, logging_wrapper: LoggingWrapper, **kwargs):
        old_loss = torch.tensor(float('inf'))
        logging_wrapper.set_num_iter(self._max_iter)
        loss = -old_loss
        it = 0

        while not self.is_converged(old_loss, loss) and it < self._max_iter:
            old_loss = loss
            loss = self._step(y)
            logging_wrapper.do_log(it, self, y)
            it += 1

        return self

    def _step(self, y) -> float:
        raise NotImplementedError()


class BatchFilterAlgorithm(BaseFilterAlgorithm, ABC):
    def __init__(self, filter_):
        """
        Implements a class of inference algorithms using filters for inference.
        """

        super().__init__(filter_)


class CombinedSequentialParticleAlgorithm(SequentialParticleAlgorithm, ABC):
    def __init__(self, filter_, particles, switch: int, first_kw, second_kw):
        """
        Algorithm combining two other algorithms.
        :param switch: After how many observations to perform switch
        """

        super().__init__(filter_, particles)
        self._first = self.make_first(filter_, particles, **(first_kw or dict()))
        self._second = self.make_second(filter_, particles, **(second_kw or dict()))
        self._when_to_switch = switch
        self._is_switched = torch.tensor(False, dtype=torch.bool)

    def make_first(self, filter_, particles, **kwargs) -> SequentialParticleAlgorithm:
        raise NotImplementedError()

    def make_second(self, filter_, particles, **kwargs) -> SequentialParticleAlgorithm:
        raise NotImplementedError()

    def do_on_switch(self, first: SequentialParticleAlgorithm, second: SequentialParticleAlgorithm):
        raise NotImplementedError()

    def initialize(self) -> BaseFilterAlgorithm:
        self._first.initialize()
        return self

    def modules(self):
        return {
            '_first': self._first,
            '_second': self._second
        }

    @property
    def logged_ess(self):
        return torch.cat((self._first.logged_ess, self._second.logged_ess))

    @property
    def _w_rec(self):
        if self._is_switched:
            return self._first._w_rec

        return self._second._w_rec

    @_w_rec.setter
    def _w_rec(self, x):
        return

    def _update(self, y: torch.Tensor) -> BaseFilterAlgorithm:
        if not self._is_switched:
            if len(self._first._logged_ess) < self._when_to_switch:
                return self._first.update(y)

            self._is_switched = True
            self.do_on_switch(self._first, self._second)

        return self._second.update(y)

    def predict(self, steps, aggregate=True, **kwargs):
        if not self._is_switched:
            return self._first.predict(steps, aggregate=aggregate, **kwargs)

        return self._second.predict(steps, aggregate=aggregate, **kwargs)
