import unittest
from pyfilter.timeseries import AffineModel, Observable, StateSpaceModel
from torch.distributions import Normal, MultivariateNormal, Independent
from pyfilter.unscentedtransform import UnscentedTransform
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
    return x[0], x[1]


def f0mvn(a, sigma):
    return torch.zeros(2)


def fomvn(x, sigma):
    return x[0] + x[1] / 2


def gomvn(x, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_UnscentedTransform1D(self):
        # ===== 1D model ===== #
        norm = Normal(0., 1.)
        linear = AffineModel((f0, g0), (f, g), (1., 1.), (norm, norm))
        linearobs = Observable((fo, go), (1., 1.), norm)
        model = StateSpaceModel(linear, linearobs)

        # ===== Perform unscented transform ===== #
        x = model.hidden.i_sample(shape=3000)

        ut = UnscentedTransform(model).initialize(x).construct(0.)

        assert isinstance(ut.x_dist, Normal)

    def test_UnscentedTransform2D(self):
        # ===== 2D model ===== #
        mat = torch.eye(2)
        scale = torch.diag(mat)

        norm = Normal(0., 1.)
        mvn = MultivariateNormal(torch.zeros(2), torch.eye(2))
        mvnlinear = AffineModel((f0mvn, g0), (fmvn, g), (mat, scale), (mvn, mvn))
        mvnoblinear = Observable((fomvn, gomvn), (1.,), norm)

        mvnmodel = StateSpaceModel(mvnlinear, mvnoblinear)

        # ===== Perform unscented transform ===== #
        x = mvnmodel.hidden.i_sample(shape=3000)

        ut = UnscentedTransform(mvnmodel).initialize(x).construct(0.)

        assert isinstance(ut.x_dist, MultivariateNormal) and isinstance(ut.y_dist, Normal)
        assert isinstance(ut.x_dist_indep, Independent)
