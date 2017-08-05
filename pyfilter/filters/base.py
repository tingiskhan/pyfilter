import pandas as pd
import numpy as np
import pyfilter.helpers.normalization as norm
from ..distributions.continuous import Distribution
import copy
from ..helpers.helpers import choose


class BaseFilter(object):
    def __init__(self, model, particles, *args, saveall=True, **kwargs):
        """
        Implements the base functionality of a particle filter.
        :param model: The state-space model to filter
        :type model: pyfilter.model.StateSpaceModel
        :param parameters:
        :param args:
        :param kwargs:
        """
        self._model = model
        self._copy = self._model.copy()

        self._particles = particles
        self._p_particles = self._particles if not isinstance(self._particles, tuple) else (self._particles[0], 1)

        self._old_y = None
        self._old_x = None
        self._old_w = None

        self.saveall = saveall
        self._td = None

        if saveall:
            self.s_x = list()
            self.s_w = list()

        self.s_l = list()

    def _initialize_parameters(self):

        # ===== HIDDEN ===== #

        self._h_params = list()
        for i, ts in enumerate(self._model.hidden):
            temp = dict()
            parameters = tuple()
            for j, p in enumerate(ts.theta):
                if isinstance(p, Distribution):
                    temp[j] = p.bounds()
                    parameters += (p.rvs(size=self._p_particles),)
                else:
                    parameters += (p,)

            ts.theta = parameters
            self._h_params.append(temp)

        # ===== OBSERVABLE ===== #

        self._o_params = dict()
        parameters = tuple()
        for j, p in enumerate(self._model.observable.theta):
            if isinstance(p, Distribution):
                self._o_params[j] = p.bounds()
                parameters += (p.rvs(size=self._p_particles),)
            else:
                parameters += (p,)

        self._model.observable.theta = parameters

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
        :return:
        """

        raise NotImplementedError()

    def longfilter(self, data):
        """
        Filters the data for the entire data set.
        :param data: An array of data. Should be {# observations, # dimensions (minimum of 1)}
        :return:
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

        w = np.array(self.s_w)
        normalized = norm.normalize(w)
        out = tuple()

        for t, xt in enumerate(self.s_x):
            weighted = tuple()
            for s in xt:
                weighted += (np.average(s, axis=-1, weights=normalized[t]),)

            out += weighted

        return out

    def predict(self, steps, **kwargs):
        """
        Predicts `steps` ahead using the latest available information.
        :param steps: The number of steps forward to predict
        :type steps: int
        :param kwargs: kwargs to pass to self._model
        :return: 
        """
        x, y = self._model.sample(steps+1, x_s=self._old_x, **kwargs)

        return x[1:], y[1:]

    def copy(self):
        """
        Returns a copy of itself.
        :return: self
        """

        return copy.deepcopy(self)

    def resample(self, indices):
        """
        Resamples the particles along the first axis.
        :param indices: The indices to choose
        :return: 
        """

        self._old_x = choose(self._old_x, indices)
        self._model.p_apply(lambda x: choose(x[0], indices))
        self._old_w = choose(self._old_w, indices)

        self.s_l = list(np.array(self.s_l)[:, indices])

        return self

    def reset(self):
        """
        Resets the filter.
        :return:
        """

        self._old_y = None
        self._old_x = self._model.initialize(self._particles)
        self._old_w = None

        if self.saveall:
            self.s_x = list()
            self.s_w = list()

        self.s_l = list()

        return self

    def replace(self, indices, ofilt):
        """
        Replaces the self instance with `ofilt`.
        :param indices: The indices to exchange
        :type indices: np.ndarray
        :param ofilt: The filter
        :type ofilt: BaseFilter
        :return:
        """

        newoldx = list()
        for tso, tsn in zip(self._old_x, ofilt._old_x):
            tso[indices] = tsn[indices]

            newoldx.append(tso)

        self._old_x = newoldx

        newsl = np.array(self.s_l)
        newsl[:, indices] = np.array(ofilt.s_l)[:, indices]

        self.s_l = list(newsl)

        return self