import unittest
from pyfilter.timeseries import (
    StateSpaceModel,
    AffineObservations,
    AffineProcess,
    models,
)
from torch.distributions import Normal, Exponential
from pyfilter.distributions import DistributionWrapper, Prior
import torch


def f(x, alpha, sigma):
    return alpha * x.values


def g(x, alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x.values


def go(x, alpha, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_Sample(self):
        norm = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        linear = AffineProcess((f, g), (1.0, 1.0), norm, norm)

        obs = AffineObservations((fo, go), (1.0, 0.0), norm)
        mod = StateSpaceModel(linear, obs)

        x, y = mod.sample_path(100)
        diff = ((x - y) ** 2).mean().sqrt()

        assert x.shape == y.shape and x.shape[0] == 100 and diff < 1e-3
