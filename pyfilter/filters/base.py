import pandas as pd
import numpy as np
import pyfilter.helpers.normalization as norm


class BaseFilter(object):
    def __init__(self, model, particles, *args, **kwargs):
        """
        Implements the base functionality of a particle filter.
        :param model: The state-space model to filter
        :type model: pyfilter.model.StateSpaceModel
        :param parameters:
        :param args:
        :param kwargs:
        """
        self._model = model
        self._particles = particles

        self._old_y = None
        self._old_x = None

        self.s_x = list()
        self.s_w = list()
        self.s_l = list()

    def initialize(self):
        """
        Initializes the filter.
        :return:
        """
        # TODO: The shape should be based on the number of particles as well as parameters if dim > 0
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

        for i in range(data.shape[0]):
            self.filter(data[i])

        return self

    def filtermeans(self):
        """
        Calculates the filter means and returns a timeseries.
        :return:
        """

        w = np.array(self.s_w)
        x = np.array(self.s_x)

        normalized = norm.normalize(w)

        weighted = tuple()
        for i in range(x.shape[1]):
            weighted += (np.average(x[:, i], axis=-1, weights=normalized),)

        return np.array(weighted).swapaxes(0, 1)