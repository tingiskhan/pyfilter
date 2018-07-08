import pandas as pd
import numpy as np
from ..distributions.continuous import Distribution
import copy
from ..utils.utils import choose, dot, expanddims
from ..utils.resampling import multinomial, systematic
from ..proposals.bootstrap import Bootstrap, Proposal
from ..timeseries import Base, StateSpaceModel


def _numparticles(parts):
    """
    Returns the correct number of particles to use
    :param parts:
    :return:
    """

    return parts if (not isinstance(parts, tuple) or (len(parts) < 2)) else (parts[0], 1)


def _overwriteparams(ts, particles):
    """
    Helper function for overwriting the parameters of the model.
    :param ts: The timeseries
    :type ts: pyfilter.timeseries.Base
    :param particles: The number of particles
    :type particles: tuple of int|int
    :return:
    """

    parambounds = dict()
    parameters = tuple()
    for j, p in enumerate(ts.theta):
        if isinstance(p, Distribution):
            parambounds[j] = p.bounds()
            parameters += (p.rvs(size=particles),)
        else:
            parameters += (p,)

    ts.theta = parameters

    return parambounds


class BaseFilter(object):
    def __init__(self, model, particles, *args, saveall=False, resampling=systematic, proposal=Bootstrap(), **kwargs):
        """
        Implements the base functionality of a particle filter.
        :param model: The state-space model to filter
        :type model: StateSpaceModel
        :param resampling: Which resampling method to use
        :type resampling: callable
        :param proposal: Which proposal to use
        :type proposal: Proposal
        :param args:
        :param kwargs:
        """
        self._model = model
        self._copy = self._model.copy()

        self._particles = particles
        self._p_particles = _numparticles(self._particles)

        self._old_x = None
        self._anc_x = None
        self._cur_x = None
        self._inds = None
        self._old_w = 0

        self._resamp = resampling

        self.saveall = saveall
        self._td = None
        self._proposal = proposal.set_model(self._model, isinstance(particles, tuple))

        if saveall:
            self.s_x = list()
            self.s_w = list()

        self.s_l = list()
        self.s_mx = list()
        self.s_n = list()

    @property
    def ssm(self):
        """
        Returns the SSM as an object.
        :rtype: StateSpaceModel
        """
        return self._model

    def _initialize_parameters(self):
        """
        Initializes the parameters by drawing from the prior distributions.
        :return:
        """

        # ===== HIDDEN ===== #

        self._h_params = _overwriteparams(self._model.hidden, self._p_particles)
        self._o_params = _overwriteparams(self._model.observable, self._p_particles)

        return self

    def initialize(self):
        """
        Initializes the filter.
        :return:
        """
        self._initialize_parameters()
        self._old_x = self._model.initialize(self._particles)

        return self

    def filter(self, y):
        """
        Filters the model for the observation `y`.
        :param y: The observation to filter on.
        :type y: float|np.ndarray
        :return:
        """

        raise NotImplementedError()

    def _calc_noise(self, y, x):
        """
        Calculates the residual given the observation `y` and state `x`.
        :param y: The observation
        :type y: np.ndarray
        :param x: The state
        :type y: np.ndarray
        :return: The residual
        :rtype: np.ndarray
        """

        mean = self._model.observable.mean(x)
        scale = self._model.observable.scale(x)

        if self._model.obs_ndim < 2:
            return (y - mean) / scale

        return dot(np.linalg.inv(scale.T).T, (expanddims(y, mean.ndim) - mean))

    def _save_mean_and_noise(self, y, x, normalized):
        """
        Saves the residual given the observation `y` and state `x`.
        :param y: The observation
        :type y: np.ndarray
        :param x: The state
        :type y: np.ndarray
        :param normalized: The normalized weights for weighting
        :type normalized: np.ndarray
        :return: Self
        :rtype: BaseFilter
        """

        rescaled = self._calc_noise(y, x)

        self.s_n.append(np.sum(rescaled * normalized, axis=-1))
        self.s_mx.append(np.sum(x * normalized, axis=-1))

        return self

    def longfilter(self, data):
        """
        Filters the data for the entire data set.
        :param data: An array of data. Should be {# observations, # dimensions (minimum of 1)}
        :return: Self
        :rtype: BaseFilter
        """

        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)

        # ===== SMC2 needs the entire dataset ==== #
        self._td = data

        for i in range(data.shape[0]):
            self.filter(data[i])

        self._td = None

        return self

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
        :return: 
        """
        x, y = self._model.sample(steps+1, x_s=self._old_x)

        return x[1:], y[1:]

    def copy(self):
        """
        Returns a copy of itself.
        :return: Copy of self
        :rtype: BaseFilter
        """

        return copy.deepcopy(self)

    def resample(self, indices):
        """
        Resamples the particles along the first axis.
        :param indices: The indices to choose
        :return: Self
        :rtype: BaseFilter
        """

        self._old_x = choose(self._old_x, indices)
        self._model.p_apply(lambda x: choose(x[0], indices))
        self._old_w = choose(self._old_w, indices)

        self._proposal = self._proposal.resample(indices)
        self.s_l = list(np.array(self.s_l)[:, indices])
        self.s_mx = list(np.array(self.s_mx)[:, indices])

        return self

    def reset(self, particles=None):
        """
        Resets the filter.
        :param particles: Size of filter to reset.
        :return: Self
        :rtype: BaseFilter
        """

        self._particles = particles if particles is not None else self._particles

        self._old_x = self._model.initialize(self._particles)
        self._old_w = 0

        if self.saveall:
            self.s_x = list()
            self.s_w = list()

        self.s_l = list()
        self.s_mx = list()

        return self

    def exchange(self, indices, newfilter):
        """
        Exchanges particles of `self` with `indices` of `newfilter`.
        :param indices: The indices to exchange
        :type indices: np.ndarray
        :param newfilter: The new filter to exchange with.
        :type newfilter: BaseFilter
        :return: Self
        :rtype: BaseFilter
        """

        # ===== Exchange parameters ===== #

        self._model.exchange(indices, newfilter._model)

        # ===== Exchange old likelihoods and weights ===== #

        for prop in ['s_l', 's_mx']:
            ots = np.array(getattr(self, prop))
            nts = np.array(getattr(newfilter, prop))

            ots[:, indices] = nts[:, indices]
            setattr(self, prop, list(ots))

        self._old_w[indices] = newfilter._old_w[indices]

        # ===== Exchange old states ===== #

        if newfilter._old_x.ndim > self._old_w.ndim:
            self._old_x[:, indices] = newfilter._old_x[:, indices]
        else:
            self._old_x[indices] = newfilter._old_x[indices]

        # ===== Exchange particle history ===== #

        if self.saveall:
            for t, (x, w) in enumerate(zip(newfilter.s_x, newfilter.s_w)):
                if x.ndim > w.ndim:
                    self.s_x[t][:, indices] = x[:, indices]
                else:
                    self.s_x[t][indices] = x[indices]

                self.s_w[t][indices] = w[indices]

        return self


class ParticleFilter(BaseFilter):
    pass


class KalmanFilter(BaseFilter):
    def exchange(self, indices, newfilter):
        """
        Exchanges particles of `self` with `indices` of `newfilter`.
        :param indices: The indices to exchange
        :type indices: np.ndarray
        :param newfilter: The new filter to exchange with.
        :type newfilter: BaseFilter
        :return: Self
        :rtype: BaseFilter
        """

        # ===== Exchange parameters ===== #

        self._model.exchange(indices, newfilter._model)

        # ===== Exchange old likelihoods and weights ===== #

        ots_l = np.array(self.s_l)
        nts_l = np.array(newfilter.s_l)

        ots_l[:, indices] = nts_l[:, indices]
        self.s_l = list(ots_l)

        # TODO: Fix this

        return self

    def resample(self, indices):
        self._model.p_apply(lambda x: choose(x[0], indices))

        self._model.p_apply(lambda x: choose(x[0], indices))
        self._proposal = self._proposal.resample(indices)
        self.s_l = list(np.array(self.s_l)[:, indices])

        return self