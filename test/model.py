import unittest

import numpy as np
import scipy.stats as stats

from pyfilter.timeseries import StateSpaceModel, Observable, BaseModel, Parameter
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
    return torch.einsum('ij,j...->i...', (a, x))


def f0mvn(a, sigma):
    return torch.zeros(2)


def fomvn(x, sigma):
    return x[0] + x[1] / 2


def gomvn(x, sigma):
    return sigma


class Tests(unittest.TestCase):
    # ===== 1D model ===== #
    norm = Normal(0., 1.)
    linear = BaseModel((f0, g0), (f, g), (1, 1), (norm, norm))
    linearobs = Observable((fo, go), (1, 1), norm)
    model = StateSpaceModel(linear, linearobs)

    # ===== 2D model ===== #
    mat = torch.eye(2)
    scale = torch.diag(mat)

    mvn = MultivariateNormal(torch.zeros(2), torch.eye(2))
    mvnlinear = BaseModel((f0mvn, g0), (fmvn, g), (mat, scale), (mvn, mvn))
    mvnoblinear = Observable((fomvn, gomvn), (1,), norm)

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

    def test_Weight(self):
        x = self.model.initialize(1000)

        y = 0

        w = self.model.weight(y, x)

        truew = stats.norm.logpdf(y, loc=x, scale=self.model.observable.theta[1])

        assert np.allclose(w, truew)

    def test_Sample(self):
        x, y = self.model.sample(50)

        assert len(x) == 50 and len(y) == 50 and np.array(x).shape == (50,)

    def test_SampleMultivariate(self):
        x, y = self.mvnmodel.sample(30)

        assert len(x) == 30 and x[0].shape == (2,)

    def test_SampleMultivariateSamples(self):
        shape = (100, 100)
        x, y = self.mvnmodel.sample(30, samples=shape)

        assert x.shape == (30, 2, *shape) and isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
        assert self.mvnmodel.h_weight(x[1], x[0]).shape == shape
        if len(shape) > 1:
            assert self.mvnmodel.h_weight(x[1, :, 0, 0], x[0]).shape == shape
        assert self.mvnmodel.weight(y[0, 0], x[0]).shape == shape

    def test_Parameter(self):
        param = Parameter(Beta(1, 3)).initialize()

        assert param.values.shape == torch.Size([])

        newshape = (3000, 1000)
        with self.assertRaises(ValueError):
            param.values = Normal(0., 1.).sample(newshape)

        newvals = Beta(1, 3).sample(newshape)
        param.values = newvals

        assert param.values.shape == newshape

        param.t_values = Normal(0., 1.).sample(newshape)

        assert (param.values != newvals).all()
