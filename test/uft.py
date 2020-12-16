import unittest
from pyfilter.timeseries import AffineProcess, AffineObservations, StateSpaceModel
from torch.distributions import Normal, MultivariateNormal
from pyfilter.uft import UnscentedFilterTransform
import torch
from pyfilter.utils import concater


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
    return concater(x[..., 0], x[..., 1])


def f0mvn(a, sigma):
    return torch.zeros(2)


def fomvn(x, sigma):
    return x[..., 0] + x[..., 1] / 2


def gomvn(x, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_UnscentedTransform1D(self):
        # ===== 1D model ===== #
        norm = Normal(0., 1.)
        linear = AffineProcess((f, g), (1., 1.), norm, norm)
        linearobs = AffineObservations((fo, go), (1., 1.), norm)
        model = StateSpaceModel(linear, linearobs)

        # ===== Perform unscented transform ===== #
        uft = UnscentedFilterTransform(model)
        res = uft.initialize(3000)
        p = uft.predict(res)
        c = uft.correct(torch.tensor(0.), p, res)

        assert isinstance(c.x_dist(), Normal) and c.x_dist().mean.shape == torch.Size([3000])

    def test_UnscentedTransform2D(self):
        # ===== 2D model ===== #
        mat = torch.eye(2)
        scale = torch.diag(mat)

        norm = Normal(0., 1.)
        mvn = MultivariateNormal(torch.zeros(2), torch.eye(2))
        mvnlinear = AffineProcess((fmvn, g), (mat, scale), mvn, mvn)
        mvnoblinear = AffineObservations((fomvn, gomvn), (1.,), norm)

        mvnmodel = StateSpaceModel(mvnlinear, mvnoblinear)

        # ===== Perform unscented transform ===== #
        uft = UnscentedFilterTransform(mvnmodel)
        res = uft.initialize(3000)
        p = uft.predict(res)
        c = uft.correct(torch.tensor(0.), p, res)

        assert isinstance(c.x_dist(), MultivariateNormal) and c.x_dist().mean.shape == torch.Size([3000, 2])

