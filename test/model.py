import unittest
import numpy as np
from pyfilter.timeseries import StateSpaceModel, Observable, AffineModel, Parameter, LinearGaussianObservations
from torch.distributions import Normal, MultivariateNormal, Beta
import torch


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f0(alpha, sigma):
    return 0


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
    # ===== 1D model ===== #
    norm = Normal(0., 1.)
    linear = AffineModel((f0, g0), (f, g), (1., 1.), (norm, norm))
    linearobs = Observable((fo, go), (1., 1.), norm)
    model = StateSpaceModel(linear, linearobs)

    # ===== 2D model ===== #
    mat = torch.eye(2)
    scale = torch.diag(mat)

    mvn = MultivariateNormal(torch.zeros(2), torch.eye(2))
    mvnlinear = AffineModel((f0mvn, g0mvn), (fmvn, gmvn), (mat, scale), (mvn, mvn))
    mvnoblinear = Observable((fomvn, gomvn), (1.,), norm)

    mvnmodel = StateSpaceModel(mvnlinear, mvnoblinear)

    def test_InitializeModel1D(self):
        sample = self.model.initialize()

        assert isinstance(sample, torch.Tensor)

    def test_InitializeModel(self):
        sample = self.model.initialize(1000)

        assert sample.shape == (1000,)

    def test_Propagate(self):
        x = self.model.initialize(1000)

        sample = self.model.propagate(x)

        assert sample.shape == (1000,)

    def test_Sample(self):
        x, y = self.model.sample(50)

        assert len(x) == 50 and len(y) == 50 and np.array(x).shape == (50,)

    def test_SampleMultivariate(self):
        x, y = self.mvnmodel.sample(30)

        assert len(x) == 30 and x[0].shape == (2,)

    def test_SampleMultivariateSamples(self):
        shape = (100, 100)
        x, y = self.mvnmodel.sample(30, samples=shape)

        assert x.shape == (30, *shape, 2) and isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)

    def test_Parameter(self):
        param = Parameter(Beta(1, 3)).sample_()

        assert param.values.shape == torch.Size([])

        newshape = (3000, 1000)
        with self.assertRaises(ValueError):
            param.values = Normal(0., 1.).sample(newshape)

        param = Parameter(Beta(1, 3)).sample_(newshape)
        param.values = Beta(1, 3).sample(newshape)

        assert param.values.shape == newshape

        newvals = Normal(0., 1.).sample(newshape)
        param.t_values = newvals

        assert (param.values == param.bijection(newvals)).all()

    def test_LinearGaussianObservations(self):
        linearmodel = LinearGaussianObservations(self.linear)

        steps = 30

        x, y = linearmodel.sample(steps)

        assert len(x) == steps and len(y) == steps

        mat = torch.eye(2)

        mvn_linearmodel = LinearGaussianObservations(self.mvnlinear, a=mat)

        x, y = mvn_linearmodel.sample(30)

        assert len(x) == steps and len(y) == steps
