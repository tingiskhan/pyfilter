import pytest
import torch

from pyfilter import inference as inf
from stochproc import timeseries as ts
from pyro.distributions import Normal, LogNormal

from pyfilter.inference.context import NotSamePriorError
from .models import build_model


BATCH_SHAPES = [torch.Size([]), torch.Size([10, 1]), torch.Size([16, 1, 1])]


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

    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_sample_parameters(self, shape):
        with inf.make_context() as context:
            alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

            assert isinstance(context.get_parameter(alpha._name), inf.PriorBoundParameter) and \
                   isinstance(context.get_parameter(beta._name), inf.PriorBoundParameter)

            context.initialize_parameters(shape)

            assert alpha.shape == shape and beta.shape == shape

    @pytest.mark.parametrize("shape", BATCH_SHAPES[1:])
    def test_exchange_context(self, shape):
        with inf.make_context() as context:
            alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

            context.initialize_parameters(shape)

            with context.make_new() as sub_context:
                alpha_sub = sub_context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
                beta_sub = sub_context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

                assert alpha_sub is not alpha and beta_sub is not beta

                sub_context.initialize_parameters(shape)

                mask: torch.BoolTensor = (torch.empty(shape[0]).normal_() > 0.0).bool()

                context.exchange(sub_context, mask)

            assert (alpha[mask] == alpha_sub[mask]).all() and (beta[mask] == beta_sub[mask]).all()

    @pytest.mark.parametrize("shape", BATCH_SHAPES[1:])
    def test_resample_context(self, shape):
        with inf.make_context() as context:
            alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

            context.initialize_parameters(shape)

            indices: torch.IntTensor = torch.randint(low=0, high=shape[0], size=shape[:1])

            alpha_clone = alpha.clone()
            beta_clone = beta.clone()
            context.resample(indices)

            assert (alpha == alpha_clone[indices]).all() and (beta == beta_clone[indices]).all()

    @pytest.mark.parametrize("shape", BATCH_SHAPES[1:])
    def test_make_model_and_resample(self, shape):
        with inf.make_context() as context:
            model = build_model(context)

            context.initialize_parameters(shape)

            old_dict = {k: v.clone() for k, v in context.parameters.items()}

            indices: torch.IntTensor = torch.randint(low=0, high=shape[0], size=shape[:1])
            context.resample(indices)

            for p_model, (n, p) in zip(model.hidden.functional_parameters(), context.get_parameters()):
                assert (p == old_dict[n][indices]).all() and (p_model is p)

    def test_assert_sampling_multiple_same(self):
        with inf.make_context() as context:
            alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

            alpha2 = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))

            with pytest.raises(NotSamePriorError):
                alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=2.0))

    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_serialize_and_load(self, shape):
        def make_model(context_):
            alpha = context_.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0))
            beta = context_.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))

            return ts.models.OrnsteinUhlenbeck(alpha, beta, beta)

        with inf.make_context() as context:
            make_model(context)

            context.initialize_parameters(shape)
            as_state = context.state_dict()

        with inf.make_context() as new_context:
            model = make_model(new_context)
            new_context.load_state_dict(as_state)

            assert (context.stack_parameters() == new_context.stack_parameters()).all()

            for p1, p2 in zip(model.functional_parameters(), new_context.parameters.values()):
                assert p1 is p2

    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_multidimensional_parameters(self, shape):
        with inf.make_context() as context:
            alpha = context.named_parameter("alpha", inf.Prior(Normal, loc=0.0, scale=1.0).expand(torch.Size([2, 2])).to_event(2))
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))
            context.initialize_parameters(shape)

            stacked = context.stack_parameters()
            assert stacked.shape == torch.Size([shape.numel(), 5])

    def test_verify_not_batched(self):
        with inf.make_context() as context:
            with pytest.raises(AssertionError):
                beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=torch.zeros(1), scale=torch.ones(1)))

    @pytest.mark.parametrize("device", ["cpu:0", "cuda:0"])
    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_apply_fun(self, device, shape):
        with inf.make_context() as context:
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0).to(device))
        
        context.initialize_parameters(shape)

        sub_context = context.apply_fun(lambda u: u.mean())
        for p, v in sub_context.parameters.items():
            assert v.shape == torch.Size([])

    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_quasi_initialize(self, shape):
        with inf.make_context(use_quasi=True) as context:
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))
            context.initialize_parameters(shape)

            assert beta.shape == shape

    @pytest.mark.parametrize("use_quasi", [False, True])
    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_copy_context(self, use_quasi, shape):
        with inf.make_context(use_quasi=use_quasi) as context:
            beta = context.named_parameter("beta", inf.Prior(LogNormal, loc=0.0, scale=1.0))
        
        context.initialize_parameters(shape)
        copy_context = context.copy()

        for (ck, cv), (k, v) in zip(copy_context.parameters.items(), context.parameters.items()):
            assert (ck == k) and (cv == v).all() and (cv.prior.equivalent_to(v.prior))
