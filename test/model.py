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


def f0(alpha, sigma):
    return torch.zeros_like(alpha)


def g0(alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x.values


def go(x, alpha, sigma):
    return sigma


def foo(x1, x2, alpha, sigma):
    return alpha * x1 + x2


def goo(x1, x2, alpha, sigma):
    return sigma


def fmvn(x, a, sigma):
    return x @ a


def f0mvn(a, sigma):
    return torch.zeros(2)


def g0mvn(a, sigma):
    return sigma * torch.ones(2)


def gmvn(x, a, sigma):
    return g0mvn(a, sigma)


def fomvn(x, sigma):
    return x[0] + x[1] / 2


def gomvn(x, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_Sample(self):
        # ==== Hidden ==== #
        norm = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        linear = AffineProcess((f, g), (1.0, 1.0), norm, norm)

        # ==== Observable ===== #
        obs = AffineObservations((fo, go), (1.0, 0.0), norm)

        # ===== Model ===== #
        mod = StateSpaceModel(linear, obs)

        # ===== Sample ===== #
        x, y = mod.sample_path(100)

        diff = ((x - y) ** 2).mean().sqrt()

        assert x.shape == y.shape and x.shape[0] == 100 and diff < 1e-3

    def test_ParametersToFromArray(self):
        priors = Prior(Exponential, rate=10.0), Prior(Normal, loc=0.0, scale=1.0), Prior(Exponential, rate=5.0)
        sde = models.OrnsteinUhlenbeck(*priors, 1, dt=1.0)

        dist = DistributionWrapper(Normal, loc=0.0, scale=Prior(Exponential, rate=5.0))
        priors = Prior(Normal, loc=0.0, scale=1.0), Prior(Exponential, rate=1.0)
        obs = AffineObservations((lambda u: u, lambda u: 1.0), priors, dist)

        mod = StateSpaceModel(sde, obs)

        mod.sample_params((100,))

        as_array = mod.parameters_to_array(constrained=False)

        assert as_array.shape == torch.Size([100, 6])

        offset = 1.0
        mod.parameters_from_array(as_array + offset, constrained=False)
        assert len(tuple(mod.parameters())) == as_array.shape[-1]

        new_array = mod.parameters_to_array()

        assert (((new_array - offset) - as_array).abs().max() < 1e-6).all()
