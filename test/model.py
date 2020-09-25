import unittest
from pyfilter.timeseries import StateSpaceModel, AffineObservations, AffineProcess, models
from torch.distributions import Normal
import torch


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f0(alpha, sigma):
    return torch.zeros_like(alpha)


def g0(alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x


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
        norm = Normal(0., 1.)
        linear = AffineProcess((f, g), (1., 1.), norm, norm)

        # ==== Observable ===== #
        obs = AffineObservations((fo, go), (1., 0.), norm)

        # ===== Model ===== #
        mod = StateSpaceModel(linear, obs)

        # ===== Sample ===== #
        x, y = mod.sample_path(100)

        diff = ((x - y) ** 2).mean().sqrt()

        assert x.shape == y.shape and x.shape[0] == 100 and diff < 1e-3

    def test_CovariateSequential(self):
        latent = models.AR(0., 0.99, 0.08)

        obs = AffineObservations((lambda u: u, lambda u: torch.tensor(0.)), (), Normal(0., 1.))
        obs.add_covariate(lambda v: v)

        x = torch.empty(101)
        y = torch.empty(x.shape[0] - 1)

        x[0] = latent.i_sample()
        for t in range(y.shape[0]):
            x[t + 1] = latent.propagate(x[t])
            y[t] = obs.propagate(x[t + 1], u=t)

        assert (((y - torch.arange(y.shape[0])) - x[1:]) < 1e-3).all()

    def test_CovariateModel(self):
        latent = models.AR(0., 0.99, 0.08)

        obs = AffineObservations((lambda u: u, lambda u: torch.tensor(0.)), (), Normal(0., 1.))
        obs.add_covariate(lambda v: v)

        mod = StateSpaceModel(latent, obs)

        covariate = torch.arange(100)
        x, y = mod.sample_path(covariate.shape[0], u=covariate)

        assert (((y - torch.arange(y.shape[0])) - x) < 1e-3).all()
