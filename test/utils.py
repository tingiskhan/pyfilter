import unittest
from pyfilter.utils import TensorTuple
import torch
from torch.distributions import Normal
from pyfilter.filters import SISR
from pyfilter.timeseries import AffineProcess, LinearGaussianObservations
from pyfilter.distributions import DistributionWrapper
from pyfilter.filters import FilterResult


class UtilTests(unittest.TestCase):
    def test_TensorList(self):
        rands = torch.empty(1000).normal_()

        tens_tuple = TensorTuple(*list(rands))

        to_load = TensorTuple()
        to_load.load_state_dict(tens_tuple.state_dict())

        self.assertTrue((to_load.values() == tens_tuple.values()).all())

    def test_LoadModule(self):
        def f(x_, alpha, sigma):
            return alpha * x_

        def g(x_, alpha, sigma):
            return sigma

        norm = DistributionWrapper(Normal, loc=0.0, scale=1.0)
        linear = AffineProcess((f, g), (1., 1.), norm, norm)
        model = LinearGaussianObservations(linear, 1., 1.)

        x, y = model.sample_path(100)

        filt = SISR(model, 200)
        res = filt.longfilter(y)

        filt2 = SISR(model, 300)

        filt2.load_state_dict(filt.state_dict())
        self.assertTrue(filt2.particles[0] == 200)

        res2 = FilterResult(filt2.initialize())
        res2.load_state_dict(res.state_dict())

        self.assertTrue((res2.filter_means == res.filter_means).all())


if __name__ == '__main__':
    unittest.main()
