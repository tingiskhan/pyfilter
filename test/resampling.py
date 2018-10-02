from pyfilter.utils.resampling import systematic
from pyfilter.utils.utils import normalize
from unittest import TestCase
import numpy as np
import torch


def filterpy_systematic_resample(weights, u):
    """
    ___NOTE___: This is the systematic resampling function from:
        https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py,
    i.e. __NOT MINE__, modified to take as input the offsetting random variable.
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (u + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


class ResamplingTests(TestCase):
    def test_Systematic1D(self):
        np.random.seed(123)

        weights = torch.tensor(np.random.normal(size=300))

        u = np.random.uniform()

        pyfilter_inds = systematic(weights, u)
        filterpy_inds = filterpy_systematic_resample(normalize(weights).numpy(), u)

        assert (pyfilter_inds.numpy() == filterpy_inds).all() and isinstance(pyfilter_inds, torch.Tensor)

    def test_Systematic2D(self):
        weights = torch.tensor(np.random.normal(size=(1000, 300)))

        u = np.random.uniform(size=(weights.shape[0], 1))

        pyfilter_inds = systematic(weights, u)

        assert isinstance(pyfilter_inds, torch.Tensor)

        for i in range(weights.shape[0]):
            filterpy_inds = filterpy_systematic_resample(normalize(weights[i]), u[i, 0])
            assert (pyfilter_inds[i].numpy() == filterpy_inds).all()

