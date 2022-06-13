import pytest
import torch

from pyfilter import inference as inf
from stochproc import timeseries as ts
from pyro.distributions import Normal, LogNormal
from .models import build_model


batch_shape = torch.Size([10, 1])


class TestContext(object):
    def test_initialize_context(self):
        context = inf.ParameterContext()

        assert len(context.stack) == 0

        with context:
            assert len(context.stack) == 1

            with context.make_new() as sub_context:
                assert (len(context.stack) == 2) and (sub_context is not context)

            assert len(context.stack) == 1

        assert len(context.stack) == 0

    def test_sample_parameters(self):
        with inf.make_context() as context:
            alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

            assert isinstance(context.get_parameter(alpha._name), inf.PriorBoundParameter) and \
                   isinstance(context.get_parameter(beta._name), inf.PriorBoundParameter)

            context.initialize_parameters(batch_shape)

            assert alpha.shape == batch_shape and beta.shape == batch_shape

    def test_exchange_context(self):
        with inf.make_context() as context:
            alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

            context.initialize_parameters(batch_shape)

            with context.make_new() as sub_context:
                alpha_sub = sub_context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
                beta_sub = sub_context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

                assert alpha_sub is not alpha and beta_sub is not beta

                sub_context.initialize_parameters(batch_shape)

                mask: torch.BoolTensor = (torch.empty(batch_shape[0]).normal_() > 0.0).bool()

                context.exchange(sub_context, mask)

            assert (alpha[mask] == alpha_sub[mask]).all() and (beta[mask] == beta_sub[mask]).all()

    def test_resample_context(self):
        with inf.make_context() as context:
            alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

            context.initialize_parameters(batch_shape)

            indices: torch.IntTensor = torch.randint(low=0, high=batch_shape[0], size=batch_shape[:1])

            alpha_clone = alpha.clone()
            beta_clone = beta.clone()
            context.resample(indices)

            assert (alpha == alpha_clone[indices]).all() and (beta == beta_clone[indices]).all()

    def test_make_model_and_resample(self):
        with inf.make_context() as context:
            model = build_model(context)

            context.initialize_parameters(batch_shape)

            old_dict = {k: v.clone() for k, v in context.parameters.items()}

            indices: torch.IntTensor = torch.randint(low=0, high=batch_shape[0], size=batch_shape[:1])
            context.resample(indices)

            for p_model, (n, p) in zip(model.hidden.functional_parameters(), context.get_parameters()):
                assert (p == old_dict[n][indices]).all() and (p_model is p)

    def test_assert_fails_register_inactive(self):
        context = inf.make_context()

        with pytest.raises(AssertionError):
            a = context.named_parameter("a", inf.Prior(Normal, loc=0.0, scale=1.0))
