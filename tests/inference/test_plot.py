import torch

from pyfilter import inference as inf
from pyro.distributions import Normal, LogNormal


class TestPlotting(object):
    def test_plotting(self):
        batch_shape = torch.Size([1_000])
        with inf.make_context() as context:
            context.set_batch_shape(batch_shape)
            alpha = context.named_parameter("alpha", Normal(loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))

            sigma = context.named_parameter(
                "sigma", LogNormal(loc=0.0, scale=1.0).expand(torch.Size([2])).to_event(1)
            )

        state = inf.sequential.state.SequentialAlgorithmState(torch.zeros(batch_shape), None)
        ax = inf.plot.mimic_arviz_posterior(context, state)

        assert ax.shape == (2, 3)
