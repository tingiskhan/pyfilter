import unittest
from pyfilter.timeseries import AffineProcess, OneStepEulerMaruyma, OrnsteinUhlenbeck, Parameter, AffineEulerMaruyama
from pyfilter.timeseries import StochasticSIR
from pyfilter.timeseries.statevariable import StateVariable
import torch
from torch.distributions import Normal, Exponential, Independent, Binomial, Poisson
import math


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f_sde(x, alpha, sigma):
    return -alpha * x


def g_sde(x, alpha, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_StateVariable(self):
        # ===== Emulate last value ===== #
        rands = torch.empty((300, 3)).normal_()
        rands.requires_grad_(True)

        # ===== Pass through function ===== #
        sv = StateVariable(rands)
        agg = sv.sum(-1)

        # ===== Get gradient ===== #
        agg.backward(torch.ones_like(agg))

        assert rands.grad is not None

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
        norm = Normal(0., 1.)
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
        norm = Normal(0., 1.)
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
        norm = Normal(0., 1.)
        shape = 1000, 100

        a = torch.ones((shape[0], 1))

        init = Normal(a, 1.)
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

        mvn = Independent(Normal(mu, scale), 1)
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

        init = Normal(a, 1.)

        dt = 1.
        norm = Normal(0., math.sqrt(dt))

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
        sde = OrnsteinUhlenbeck(a, 0., 0.15, 1, dt=1.)

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
        dist = Normal(loc=0., scale=Parameter(Exponential(10.)))

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

    def test_StochasticSIR(self):
        dist = Independent(Binomial(torch.tensor([1000, 1, 0]), torch.tensor([1, 1, 1e-6])), 1)
        sir = StochasticSIR((0.1, 0.05, 0.01), dist, 1e-1)

        x = sir.sample_path(1000, 10)

        assert x.shape == torch.Size([1000, 10, 3])
