import unittest
from pyfilter.timeseries import AffineProcess, AffineObservations, StateSpaceModel
from torch.distributions import Normal, MultivariateNormal
from pyfilter.filters.kalman.unscented import UnscentedFilterTransform
import torch
from pyfilter.utils import concater
from pyfilter.distributions import DistributionWrapper


def f(x, alpha, sigma):
    return alpha * x.state


def g(x, alpha, sigma):
    return sigma


def f0(alpha, sigma):
    return 0


def g0(alpha, sigma):
    return sigma


def f_mv(x, a, sigma):
    return concater(x.state[..., 0], x.state[..., 1])


def f0_mv(a, sigma):
    return torch.zeros(2)


def fo_mv(x, sigma):
    return x.state[..., 0] + x.state[..., 1] / 2


def go_mv(x, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_UnscentedTransform1D(self):
        norm = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        linear = AffineProcess((f, g), (1.0, 1.0), norm, norm)
        linearobs = AffineObservations((f, g), (1.0, 1.0), norm)
        model = StateSpaceModel(linear, linearobs)

        uft = UnscentedFilterTransform(model)
        res = uft.initialize(3000)
        p = uft.predict(res)
        c = uft.correct(torch.tensor(0.0), p, res)

        assert isinstance(c.x_dist(), Normal) and c.x_dist().mean.shape == torch.Size([3000])

    def test_UnscentedTransform2D(self):
        mat = torch.eye(2)
        scale = torch.diag(mat)

        norm = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        mvn = DistributionWrapper(MultivariateNormal, loc=torch.zeros(2), covariance_matrix=torch.eye(2))

        mvnlinear = AffineProcess((f_mv, g), (mat, scale), mvn, mvn)
        mvnoblinear = AffineObservations((fo_mv, go_mv), (1.0,), norm)

        mvnmodel = StateSpaceModel(mvnlinear, mvnoblinear)

        uft = UnscentedFilterTransform(mvnmodel)
        res = uft.initialize(3000)
        p = uft.predict(res)
        c = uft.correct(torch.tensor(0.0), p, res)

        assert isinstance(c.x_dist(), MultivariateNormal) and c.x_dist().mean.shape == torch.Size([3000, 2])
