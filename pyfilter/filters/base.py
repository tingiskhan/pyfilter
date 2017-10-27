import pandas as pd
import numpy as np
from ..distributions.continuous import Distribution
import copy
from ..utils.utils import choose
from ..utils.resampling import multinomial, systematic
from ..proposals.bootstrap import Bootstrap


class BaseFilter(object):
    def __init__(self, model, particles, *args, saveall=False, resampling=systematic, proposal=Bootstrap, **kwargs):
        """
        Implements the base functionality of a particle filter.
        :param model: The state-space model to filter
        :type model: pyfilter.model.StateSpaceModel
        :param resampling: Which resampling method to use
        :type resampling: function
        :param args:
        :param kwargs:
        """
        self._model = model
        self._copy = self._model.copy()

        self._particles = particles
        self._p_particles = self._particles if not isinstance(self._particles, tuple) else (self._particles[0], 1)

        self._old_x = None
        self._anc_x = None
        self._cur_x = None
        self._inds = None
        self._old_w = 0

        self._resamp = resampling

        self.saveall = saveall
        self._td = None
        self._proposal = proposal(self._model, isinstance(particles, tuple))

        if saveall:
            self.s_x = list()
            self.s_w = list()

        self.s_l = list()
        self.s_mx = list()

    def _initialize_parameters(self):

        # ===== HIDDEN ===== #

        self._h_params = dict()
        parameters = tuple()
        for j, p in enumerate(self._model.hidden.theta):
            if isinstance(p, Distribution):
                self._h_params[j] = p.bounds()
                parameters += (p.rvs(size=self._p_particles),)
            else:
                parameters += (p,)

        self._model.hidden.theta = parameters

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

        return self.s_mx

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

    def reset(self, particles=None):
        """
        Resets the filter.
        :param particles: Size of filter to reset.
        :return:
        """

        self._particles = particles if particles is not None else self._particles

        self._old_x = self._model.initialize(self._particles)
        self._old_w = 0

        if self.saveall:
            self.s_x = list()
            self.s_w = list()

        self.s_l = list()

        return self

    def exchange(self, indices, newfilter):
        """
        Exchanges particles of `self` with `indices` of `newfilter`.
        :param indices: The indices to exchange
        :type indices: np.ndarray
        :param newfilter: The new filter to exchange with.
        :type newfilter: BaseFilter
        :return:
        """

        # ===== Exchange parameters ===== #

        self._model.exchange(indices, newfilter._model)

        # ===== Exchange old likelihoods and weights ===== #

        ots_l = np.array(self.s_l)
        nts_l = np.array(newfilter.s_l)

        ots_l[:, indices] = nts_l[:, indices]
        self.s_l = list(ots_l)
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