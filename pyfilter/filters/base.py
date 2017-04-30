import pandas as pd
import numpy as np


class BaseFilter(object):
    def __init__(self, model, parameters, particles, *args, **kwargs):
        """
        Implements the base functionality of a particle filter
        :param model:
        :param parameters:
        :param args:
        :param kwargs:
        """
        self._model = model
        self.parameters = parameters
        self._particles = particles

        self._old_y = None
        self._old_x = None

    def initialize(self):
        """
        Initializes the filter.
        :return:
        """

        self._old_x = self._model.sample_initial(self.parameters)

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
