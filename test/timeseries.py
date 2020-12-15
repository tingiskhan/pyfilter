import unittest
from pyfilter.timeseries import (
    AffineProcess,
    OneStepEulerMaruyma,
    Parameter,
    AffineEulerMaruyama,
    models as m
)
import torch
from torch.distributions import Normal, Exponential, Independent, Binomial, Poisson, Dirichlet
import math
from pyfilter.distributions import DistributionWrapper


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f_sde(x, alpha, sigma):
    return -alpha * x


def g_sde(x, alpha, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_Parameter(self):
        # ===== Start stuff ===== #
        param = Parameter(Normal(0., 1.))
        param.sample_(1000)

        assert param.shape == torch.Size([1000])

        # ===== Construct view ===== #
        view = param.view(1000, 1)

        # ===== Change values ===== #
        param.values = torch.empty_like(param).normal_()

        assert (view[:, 0] == param).all()

        # ===== Have in tuple ===== #
        vals = (param.view(1000, 1, 1),)

        param.values = torch.empty_like(param).normal_()

        assert (vals[0][:, 0, 0] == param).all()

        # ===== Set t_values ===== #
        view = param.view(1000, 1)

        param.t_values = torch.empty_like(param).normal_()

        assert (view[:, 0] == param.t_values).all()

        # ===== Check we cannot set different shape ===== #
        with self.assertRaises(ValueError):
            param.values = torch.empty(1).normal_()

        # ===== Check that we cannot set out of bounds values for parameter ===== #
        positive = Parameter(Exponential(1.))
        positive.sample_(1)

        with self.assertRaises(ValueError):
            positive.values = -torch.empty_like(positive).normal_().abs()

        # ===== Check that we can set transformed values ===== #
        values = torch.empty_like(positive).normal_()
        positive.t_values = values

        assert (positive == positive.bijection(values)).all()

    def test_LinearNoBatch(self):
        norm = DistributionWrapper(Normal, loc=0., scale=1.)
        linear = AffineProcess((f, g), (1., 1.), norm, norm)

        # ===== Initialize ===== #
        x = linear.i_sample()

        # ===== Propagate ===== #
        num = 100
        samps = [x]
        for t in range(num):
            samps.append(linear.propagate(samps[-1]))

        samps = torch.stack(samps)
        self.assertEqual(samps.size(), torch.Size([num + 1]))

        # ===== Sample path ===== #
        path = linear.sample_path(num + 1)
        self.assertEqual(samps.shape, path.shape)

    def test_LinearBatch(self):
        norm = DistributionWrapper(Normal, loc=0., scale=1.)
        linear = AffineProcess((f, g), (1., 1.), norm, norm)

        # ===== Initialize ===== #
        shape = 1000, 100
        x = linear.i_sample(shape)

        # ===== Propagate ===== #
        num = 100
        samps = [x]
        for t in range(num):
            samps.append(linear.propagate(samps[-1]))

        samps = torch.stack(samps)
        self.assertEqual(samps.size(), torch.Size([num + 1, *shape]))

        # ===== Sample path ===== #
        path = linear.sample_path(num + 1, shape)
        self.assertEqual(samps.shape, path.shape)

    def test_BatchedParameter(self):
        norm = DistributionWrapper(Normal, loc=0., scale=1.)
        shape = 1000, 100

        a = torch.ones((shape[0], 1))

        init = DistributionWrapper(Normal, loc=a, scale=1.)
        linear = AffineProcess((f, g), (a, 1.), init, norm)

        # ===== Initialize ===== #
        x = linear.i_sample(shape)

        # ===== Propagate ===== #
        num = 100
        samps = [x]
        for t in range(num):
            samps.append(linear.propagate(samps[-1]))

        samps = torch.stack(samps)
        self.assertEqual(samps.size(), torch.Size([num + 1, *shape]))

        # ===== Sample path ===== #
        path = linear.sample_path(num + 1, shape)
        self.assertEqual(samps.shape, path.shape)

    def test_MultiDimensional(self):
        mu = torch.zeros(2)
        scale = torch.ones_like(mu)

        shape = 1000, 100

        mvn = DistributionWrapper(lambda **u: Independent(Normal(**u), 1), loc=mu, scale=scale)
        mvn = AffineProcess((f, g), (1., 1.), mvn, mvn)

        # ===== Initialize ===== #
        x = mvn.i_sample(shape)

        # ===== Propagate ===== #
        num = 100
        samps = [x]
        for t in range(num):
            samps.append(mvn.propagate(samps[-1]))

        samps = torch.stack(samps)
        self.assertEqual(samps.size(), torch.Size([num + 1, *shape, *mu.shape]))

        # ===== Sample path ===== #
        path = mvn.sample_path(num + 1, shape)
        self.assertEqual(samps.shape, path.shape)

    def test_OneStepEuler(self):
        shape = 1000, 100

        a = 1e-2 * torch.ones((shape[0], 1))

        init = DistributionWrapper(Normal, loc=a, scale=1.)

        dt = 1.
        norm = DistributionWrapper(Normal, loc=0., scale=math.sqrt(dt))

        sde = OneStepEulerMaruyma((f_sde, g_sde), (a, 0.15), init, norm, dt)

        # ===== Initialize ===== #
        x = sde.i_sample(shape)

        # ===== Propagate ===== #
        num = 100
        samps = [x]
        for t in range(num):
            samps.append(sde.propagate(samps[-1]))

        samps = torch.stack(samps)
        self.assertEqual(samps.size(), torch.Size([num + 1, *shape]))

        # ===== Sample path ===== #
        path = sde.sample_path(num + 1, shape)
        self.assertEqual(samps.shape, path.shape)

    def test_OrnsteinUhlenbeck(self):
        shape = 1000, 100

        a = 1e-2 * torch.ones((shape[0], 1))
        sde = m.OrnsteinUhlenbeck(a, 0., 0.15, 1, dt=1.)

        # ===== Initialize ===== #
        x = sde.i_sample(shape)

        # ===== Propagate ===== #
        num = 100
        samps = [x]
        for t in range(num):
            samps.append(sde.propagate(samps[-1]))

        samps = torch.stack(samps)
        self.assertEqual(samps.size(), torch.Size([num + 1, *shape]))

        # ===== Sample path ===== #
        path = sde.sample_path(num + 1, shape)
        self.assertEqual(samps.shape, path.shape)

    def test_SDE(self):
        shape = 1000, 100

        a = 1e-2 * torch.ones((shape[0], 1))
        dt = 0.1
        norm = Normal(0., math.sqrt(dt))

        init = Normal(a, 1.)
        sde = AffineEulerMaruyama((f_sde, g_sde), (a, 0.15), init, norm, dt=dt, num_steps=10)

        # ===== Initialize ===== #
        x = sde.i_sample(shape)

        # ===== Propagate ===== #
        num = 100
        samps = [x]
        for t in range(num):
            samps.append(sde.propagate(samps[-1]))

        samps = torch.stack(samps)
        self.assertEqual(samps.size(), torch.Size([num + 1, *shape]))

        # ===== Sample path ===== #
        path = sde.sample_path(num + 1, shape)
        self.assertEqual(samps.shape, path.shape)

    def test_Poisson(self):
        shape = 10, 100

        a = 1e-2 * torch.ones((shape[0], 1))
        dt = 1e-2
        dist = Poisson(dt * 0.1)

        init = Normal(a, 1.)
        sde = AffineEulerMaruyama((f_sde, g_sde), (a, 0.15), init, dist, dt=dt, num_steps=10)

        # ===== Initialize ===== #
        x = sde.i_sample(shape)

        # ===== Propagate ===== #
        num = 1000
        samps = [x]
        for t in range(num):
            samps.append(sde.propagate(samps[-1]))

        samps = torch.stack(samps)
        self.assertEqual(samps.size(), torch.Size([num + 1, *shape]))

        # ===== Sample path ===== #
        path = sde.sample_path(num + 1, shape)
        self.assertEqual(samps.shape, path.shape)

    def test_ParameterInDistribution(self):
        shape = 10, 100

        a = 1e-2 * torch.ones((shape[0], 1))
        dt = 1e-2
        dist = DistributionBuilder(Normal, loc=0., scale=Parameter(Exponential(10.)))

        init = Normal(a, 1.)
        sde = AffineEulerMaruyama((f_sde, g_sde), (a, 0.15), init, dist, dt=dt, num_steps=10)

        sde.sample_params(shape)

        # ===== Initialize ===== #
        x = sde.i_sample(shape)

        # ===== Propagate ===== #
        num = 1000
        samps = [x]
        for t in range(num):
            samps.append(sde.propagate(samps[-1]))

        samps = torch.stack(samps)
        self.assertEqual(samps.size(), torch.Size([num + 1, *shape]))

        # ===== Sample path ===== #
        path = sde.sample_path(num + 1, shape)
        self.assertEqual(samps.shape, path.shape)

    def test_AR(self):
        ar = m.AR(0., 0.99, 0.08)

        x = ar.sample_path(100)

        self.assertEqual(x.shape, torch.Size([100]))

    def test_ParametersToFromArray(self):
        sde = m.OrnsteinUhlenbeck(Exponential(10.), Normal(0., 1.), Exponential(5.), 1, dt=1.)
        sde.sample_params(100)

        as_array = sde.parameters_to_array(transformed=False)

        assert as_array.shape == torch.Size([100, 3])

        offset = 1.
        sde.parameters_from_array(as_array + offset, transformed=False)
        assert len(sde.parameters) == as_array.shape[-1]

        for i, p in enumerate(sde.trainable_parameters):
            assert (((p - offset) - as_array[:, i]).abs() < 1e-6).all()
