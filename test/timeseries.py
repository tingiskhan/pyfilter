import unittest
from pyfilter.timeseries import AffineProcess, OneStepEulerMaruyma, AffineEulerMaruyama, models as m
import torch
from torch.distributions import (
    Normal,
    Exponential,
    Independent,
    Poisson,
    Distribution,
    TransformedDistribution,
    AffineTransform,
)
import math
from pyfilter.distributions import DistributionWrapper, Prior
from pyfilter.timeseries import StochasticProcess
from pyfilter.timeseries.state import NewState
from pyfilter.typing import ShapeLike


def f(x, alpha, sigma):
    return alpha * x.values


def g(x, alpha, sigma):
    return sigma


def f_sde(x, alpha, sigma):
    return -alpha * x.values


def g_sde(x, alpha, sigma):
    return sigma


class CustomModel(StochasticProcess):
    def build_density(self, x: NewState) -> Distribution:
        return Normal(loc=x.values, scale=torch.ones_like(x.values))


def build_model(initial_transform=None):
    norm = DistributionWrapper(Normal, loc=0.0, scale=1.0)
    return AffineProcess((f, g), (1.0, 1.0), norm, norm, initial_transform=initial_transform)


class TimeseriesTests(unittest.TestCase):
    def assert_timeseries_sampling(self, steps, model, initial, shape, expected_shape=None):
        samps = [initial]
        for t in range(steps):
            samps.append(model.propagate(samps[-1]))

        samps = torch.stack(tuple(s.values for s in samps))
        self.assertEqual(samps.size(), torch.Size([steps + 1, *(expected_shape or shape)]))

        path = model.sample_path(steps + 1, shape)
        self.assertEqual(samps.shape, path.shape)

    def test_LinearNoBatch(self):
        custom_model = CustomModel(initial_dist=DistributionWrapper(Normal, loc=0.0, scale=1.0))

        x = custom_model.initial_sample()

        self.assert_timeseries_sampling(100, custom_model, x, ())

    def test_LinearBatch(self):
        linear = build_model()

        shape = 1000, 100
        x = linear.initial_sample(shape)

        self.assert_timeseries_sampling(100, linear, x, shape)

    def test_BatchedParameter(self):
        norm = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        shape = 1000, 100

        a = torch.ones((shape[0], 1))

        init = DistributionWrapper(Normal, loc=a, scale=1.0)
        linear = AffineProcess((f, g), (a, 1.0), init, norm)

        x = linear.initial_sample(shape)
        self.assert_timeseries_sampling(100, linear, x, shape)

    def test_MultiDimensional(self):
        mu = torch.zeros(2)
        scale = torch.ones_like(mu)

        shape = 1000, 100

        mvn = DistributionWrapper(lambda **u: Independent(Normal(**u), 1), loc=mu, scale=scale)
        mvn = AffineProcess((f, g), (1.0, 1.0), mvn, mvn)

        x = mvn.initial_sample(shape)
        self.assert_timeseries_sampling(100, mvn, x, shape, (*shape, 2))

    def test_PriorWithDistribution(self):
        mu = Prior(Normal, loc=0.0, scale=1.0)
        shape = ()

        def initial_transform(module: AffineProcess, dist):
            params = module.functional_parameters()
            return TransformedDistribution(dist, AffineTransform(*params))

        norm = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        model = AffineProcess((f, g), (mu, 1.0), norm, norm, initial_transform=initial_transform)

        x = model.initial_sample(shape)

        self.assertTrue(isinstance(x.dist, TransformedDistribution))
        self.assert_timeseries_sampling(100, model, x, shape)

    def test_OneStepEuler(self):
        shape = 1000, 100

        a = 1e-2 * torch.ones((shape[0], 1))

        init = DistributionWrapper(Normal, loc=a, scale=1.0)

        dt = 1.0
        norm = DistributionWrapper(Normal, loc=0.0, scale=math.sqrt(dt))

        sde = OneStepEulerMaruyma((f_sde, g_sde), (a, 0.15), init, norm, dt)

        x = sde.initial_sample(shape)
        self.assert_timeseries_sampling(100, sde, x, shape)

    def test_OrnsteinUhlenbeck(self):
        shape = 1000, 100

        a = 1e-2 * torch.ones((shape[0], 1))
        sde = m.OrnsteinUhlenbeck(a, 0.0, 0.15, 1, dt=1.0)

        x = sde.initial_sample(shape)
        self.assert_timeseries_sampling(100, sde, x, shape)

    def test_SDE(self):
        shape = 1000, 100

        a = 1e-2 * torch.ones((shape[0], 1))
        dt = 0.1

        init = norm = DistributionWrapper(Normal, loc=a, scale=math.sqrt(dt))
        sde = AffineEulerMaruyama((f_sde, g_sde), (a, 0.15), init, norm, dt=dt, num_steps=10)

        x = sde.initial_sample(shape)
        self.assert_timeseries_sampling(100, sde, x, shape)

    def test_ParameterInDistribution(self):
        shape = 10, 100

        dt = 1e-2
        dist = DistributionWrapper(Normal, loc=0.0, scale=Prior(Exponential, rate=10.0))

        init = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        sde = AffineEulerMaruyama((f_sde, g_sde), (1.0, 0.15), init, dist, dt=dt, num_steps=10)

        sde.sample_params(shape)

        x = sde.initial_sample(shape)
        self.assert_timeseries_sampling(100, sde, x, shape)

    def test_AR(self):
        ar = m.AR(0.0, 0.99, 0.08)

        x = ar.sample_path(100)
        self.assertEqual(x.shape, torch.Size([100]))
