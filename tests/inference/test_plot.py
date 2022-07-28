import torch

from pyfilter import inference as inf
from pyro.distributions import Normal, LogNormal


class TestPlotting(object):
    def test_plotting(self):
        batch_shape = torch.Size([1_000])
        with inf.make_context() as context:
            alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

            sigma = context.named_parameter(
                "sigma", inf.Prior(LogNormal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1)
            )

            context.initialize_parameters(batch_shape)

        state = inf.sequential.state.SequentialAlgorithmState(torch.zeros(batch_shape), None)
        ax = inf.plot.mimic_arviz_posterior(context, state)

        assert ax.shape == (2, 3)
