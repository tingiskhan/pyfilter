import pytest
import torch

import pyfilter.inference as inf
from pyro.distributions import Normal
from stochproc.timeseries import StructuralStochasticProcess


SHAPES = [
    torch.Size([]),
    torch.Size([200, 1])
]


class TestParameter(object):
    @pytest.mark.parametrize("batch_shape", SHAPES)
    def test_initialize_parameter(self, batch_shape):
        with inf.make_context() as cntxt:
            prior = inf.Prior(Normal, loc=0.0, scale=1.0)
            parameter = cntxt.named_parameter("kappa", prior)

            sts = StructuralStochasticProcess((parameter,), None)
            assert (next(sts.parameters()) is parameter) and (cntxt.get_parameter("kappa") is parameter)

            for p in sts.parameters():
                p.sample_(batch_shape)
                assert (p is parameter) and (p.shape == batch_shape)

            if not torch.cuda.is_available():
                return

            sts = sts.cuda()
            assert (next(sts.parameters()) is parameter) and (cntxt.get_parameter("kappa") is parameter)

            for p in sts.parameters():
                p.sample_(batch_shape)
                assert (p is parameter) and (p.shape == batch_shape)

    @pytest.mark.parametrize("batch_shape", SHAPES)
    def test_sample_by_inversion(self, batch_shape):
        with inf.make_context() as cntxt:
            prior = inf.Prior(Normal, loc=0.0, scale=1.0)
            parameter = cntxt.named_parameter("kappa", prior)

            se = torch.quasirandom.SobolEngine(parameter.prior.shape.numel(), scramble=True)
            samples = se.draw(batch_shape.numel())

            parameter.inverse_sample_(samples)

            assert parameter.shape == samples.shape
            assert ((parameter - prior.build_distribution().icdf(samples)) == 0.0).all()
