import pandas as pd
import numpy as np
import copy
from ..utils.utils import choose, dot, expanddims
from ..utils.resampling import multinomial, systematic
from ..proposals.bootstrap import Bootstrap, Proposal
from ..timeseries import BaseModel, StateSpaceModel
from tqdm import tqdm
import torch


class BaseFilter(object):
    def __init__(self, model):
        """
        The basis for filters. Take as input a model and specific attributes.
        :param model: The model
        :type model: StateSpaceModel
        """

        if not isinstance(model, StateSpaceModel):
            raise ValueError('`model` must be `{:s}`!'.format(StateSpaceModel.__name__))

        self._model = model
        self._td = None     # type: np.ndarray

        # ===== Some helpers ===== #
        self.s_ll = tuple()
        self.s_mx = tuple()
        self.s_n = tuple()

    @property
    def ssm(self):
        """
        Returns the SSM as an object.
        :rtype: StateSpaceModel
        """
        return self._model

    def initialize(self):
        """
        Initializes the filter.
        :return: Self
        :rtype: BaseFilter
        """

        return self

    def filter(self, y):
        """
        Performs a filtering the model for the observation `y`.
        :param y: The observation
        :type y: float|np.ndarray
        :return: Self
        :rtype: BaseFilter
        """

        xm, ll = self._filter(y)

        self.s_mx += (xm,)
        self.s_ll += (ll,)

        return self

    def _filter(self, y):
        """
        The actual filtering procedure. Overwrite this.
        :param y: The observation
        :type y: float|np.ndarray
        :return: Mean of state, likelihood
        :rtype: torch.Tensor, torch.Tensor
        """

        raise NotImplementedError()

    def longfilter(self, y, bar=True):
        """
        Filters the entire data set `y`.
        :param y: An array of data. Should be {# observations, # dimensions (minimum of 1)}
        :type y: pd.DataFrame|np.ndarray
        :param bar: Whether to print a progressbar
        :type bar: bool
        :return: Self
        :rtype: BaseFilter
        """

        if isinstance(y, pd.DataFrame):
            y = y.values

        # ===== Make a temporary copy of the data ==== #
        self._td = y

        if bar:
            iterator = tqdm(range(y.shape[0]), desc=str(self.__class__.__name__))
        else:
            iterator = range(y.shape[0])

        for i in iterator:
            self.filter(y[i])

        self._td = None

        return self

    def copy(self):
        """
        Returns a copy of itself.
        :return: Copy of self
        :rtype: BaseFilter
        """

        return copy.deepcopy(self)

    def filtermeans(self):
        """
        Calculates the filter means and returns a timeseries.
        :return:
        """

        return self.s_mx

    def noisemeans(self):
        """
        Calculates the means for the noise and returns a timeseries.
        :return:
        """

        return self.s_n

    def predict(self, steps):
        """
        Predicts `steps` ahead using the latest available information.
        :param steps: The number of steps forward to predict
        :type steps: int
        :rtype: tuple[torch.Tensor]
        """
        raise NotImplementedError()

    def resample(self, inds, entire_history=False):
        """
        Resamples the filter, used in cases where we use nested filters.
        :param inds: The indices
        :type inds: torch.Tensor
        :param entire_history: Whether to resample entire history
        :type entire_history: bool
        :return: Self
        :rtype: BaseFilter
        """
        # TODO: Fix this
        if entire_history:
            for obj in [self.s_mx, self.s_ll]:
                as_tens = torch.Tensor(obj)

        # ===== Resample the parameters of the model ====== #
        self.ssm.p_apply(lambda u: u.values[..., inds, :])
        self._resample(inds)

        return self

    def _resample(self, inds):
        """
        Implements resampling unique for the filter.
        :param inds: The indices
        :type inds: torch.Tensor
        :return: Self
        :rtype: BaseFilter
        """

        return self


class ParticleFilter(BaseFilter):
    def __init__(self, model, particles, resampling=systematic, proposal=Bootstrap()):
        """
        Implements the base functionality of a particle filter.
        :param particles: How many particles to use
        :type particles: int
        :param resampling: Which resampling method to use
        :type resampling: callable
        :param proposal: Which proposal to use
        :type proposal: Proposal
        """

        super().__init__(model)

        self._x_cur = None                          # type: torch.Tensor
        self._inds = None                           # type: torch.Tensor
        self._particles = particles
        self._w_old = torch.zeros(self._particles)  # type: torch.Tensor

        self._resamp = resampling

        self._proposal = proposal.set_model(self._model, isinstance(particles, tuple))

    def set_particles(self, x):
        """
        Sets the particles.
        :param x: The particles
        :type x: int|tuple[int]
        :return: Self
        :rtype: ParticleFilter
        """

        self._particles = x

        if self._x_cur is not None:
            return self.initialize()

        self._w_old = torch.zeros(self._particles)

        return self

    def initialize(self):
        self._x_cur = self._model.initialize(self._particles)

        return self

    def predict(self, steps):
        x, y = self._model.sample(steps+1, x_s=self._x_cur)

        return x[1:], y[1:]

    def _resample(self, inds):
        self._x_cur = self._x_cur[..., inds, :]
        self._w_old = self._w_old[inds]

        return self


class KalmanFilter(BaseFilter):
    pass