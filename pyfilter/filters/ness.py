from .base import BaseFilter
from .bootstrap import Bootstrap
import pandas as pd
import numpy as np


class NESS(object):
    def __init__(self, model, particles, *args, filt=Bootstrap, **kwargs):
        """
        Implements the NESS algorithm https://arxiv.org/abs/1308.1883.
        :param model: The state-space model to filter
        :type model: pyfilter.model.StateSpaceModel
        :param particles: The number of particles to use
        :type particles: tuple
        :param filt: The filter to use
        :type filt: BaseFilter
        :param args:
        :param kwargs:
        """

        assert isinstance(particles, tuple)

        self._filt = filt(model, particles, *args, **kwargs).initialize()
        self._c_filt =

    def filter(self, y):
        pass

    def run(self, data):
        """
        Runs the algorithm.
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