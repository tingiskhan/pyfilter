import numpy as np
from pytest import fixture
import torch
from pyfilter.resampling import systematic
from pyfilter.utils import normalize


def filterpy_systematic_resample(weights, u):
    """
    ___NOTE___: This is the systematic resampling function from:
        https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py,
    i.e. __NOT MINE__, modified to take as input the offsetting random variable.
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (u + np.arange(N)) / N

    indexes = np.zeros(N)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


torch.random.manual_seed(123)


@fixture
def weights():
    return normalize(torch.randn((10, 300), dtype=torch.float64))


class TestResampling(object):
    def test_systematic(self, weights):        
        u = torch.rand(weights.shape)

        pyfilter_inds = systematic(weights.moveaxis(0, 1), u=u, normalized=True).moveaxis(0, 1).numpy()

        for i in range(weights.shape[0]):
            filterpy_inds = filterpy_systematic_resample(weights[i], u[i])
            assert (pyfilter_inds[i] == filterpy_inds).all()
