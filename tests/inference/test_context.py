import pytest
import torch

from pyfilter import inference as inf
from stochproc import timeseries as ts
from pyro.distributions import Normal, LogNormal
from concurrent import futures

from pyfilter.inference.context import NotSamePriorError


BATCH_SHAPES = [torch.Size([]), torch.Size([10]), torch.Size([1, 1, 12])]


class TestContext(object):
    def test_initialize_context(self):
        context = inf.InferenceContext()

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
            context.set_batch_shape(shape)

            alpha = context.named_parameter("alpha", Normal(loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))

            assert isinstance(context.get_parameter(alpha._name), inf.PriorBoundParameter) and \
                   isinstance(context.get_parameter(beta._name), inf.PriorBoundParameter)

            assert alpha.shape == shape and beta.shape == shape

    @pytest.mark.parametrize("shape", BATCH_SHAPES[1:])
    def test_exchange_context(self, shape):
        with inf.make_context() as context:
            context.set_batch_shape(shape)

            alpha = context.named_parameter("alpha", Normal(loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))

            context.initialize_parameters()

            with context.make_new() as sub_context:
                sub_context.set_batch_shape(shape)
                
                alpha_sub = sub_context.named_parameter("alpha", Normal(loc=0.0, scale=1.0))
                beta_sub = sub_context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))

                assert alpha_sub is not alpha and beta_sub is not beta
                
                context.initialize_parameters()

                mask_has_trues = False
                while not mask_has_trues:
                    mask = (torch.empty(shape[0]).normal_() > 0.0).bool()
                    mask_has_trues = mask.any()

                context.exchange(sub_context, mask)

            assert (alpha[mask] == alpha_sub[mask]).all() and (beta[mask] == beta_sub[mask]).all()

    @pytest.mark.parametrize("shape", BATCH_SHAPES[1:])
    def test_resample_context(self, shape):
        with inf.make_context() as context:
            context.set_batch_shape(shape)

            alpha = context.named_parameter("alpha", Normal(loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))
            
            context.initialize_parameters()

            indices: torch.IntTensor = torch.randint(low=0, high=shape[0], size=shape[:1])

            alpha_clone = alpha.clone()
            beta_clone = beta.clone()
            context.resample(indices)

            assert (alpha == alpha_clone[indices]).all() and (beta == beta_clone[indices]).all()

    def test_assert_sampling_multiple_same(self):
        with inf.make_context() as context:
            context.set_batch_shape(torch.Size([]))

            alpha = context.named_parameter("alpha", Normal(loc=0.0, scale=1.0))
            beta = context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))

            alpha2 = context.named_parameter("alpha", Normal(loc=0.0, scale=1.0))

            assert alpha is alpha2

            with pytest.raises(NotSamePriorError):
                alpha = context.named_parameter("alpha", Normal(loc=0.0, scale=2.0))

    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_serialize_and_load(self, shape):
        def make_model(context_):
            alpha = context_.named_parameter("alpha", Normal(loc=0.0, scale=1.0))
            beta = context_.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))

            return ts.models.OrnsteinUhlenbeck(alpha, beta, beta)

        with inf.make_context() as context:
            context.set_batch_shape(shape)
            make_model(context)

            as_state = context.state_dict()

        with inf.make_context() as new_context:
            new_context.set_batch_shape(shape)
            model = make_model(new_context)
            new_context.load_state_dict(as_state)

            assert (context.stack_parameters() == new_context.stack_parameters()).all()

            for p1, p2 in zip(model.parameters, new_context.parameters.values()):
                assert p1 is not p2

    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    @pytest.mark.parametrize("constrained", [True, False])
    def test_multidimensional_parameters(self, shape, constrained):
        with inf.make_context() as context:
            context.set_batch_shape(shape)

            alpha = context.named_parameter("alpha", Normal(loc=0.0, scale=1.0).expand(torch.Size([2, 2])).to_event(2))
            beta = context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))

            stacked = context.stack_parameters(constrained)
            assert stacked.shape == torch.Size([shape.numel(), 5])

    def test_verify_not_batched(self):
        with inf.make_context() as context:
            context.set_batch_shape(torch.Size([]))
            with pytest.raises(AssertionError):
                beta = context.named_parameter("beta", LogNormal(loc=torch.zeros(1), scale=torch.ones(1)))

    @pytest.mark.parametrize("device", ["cpu:0", "cuda:0"])
    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_apply_fun(self, device, shape):
        if (device == "cuda:0") and not torch.cuda.is_available():
            return

        with inf.make_context() as context:
            context.set_batch_shape(shape)
            beta = context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0).to(device))

        sub_context = context.apply_fun(lambda u: u.mean())
        for p, v in sub_context.parameters.items():
            assert (v == context.get_parameter(p).mean()).all()

    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_quasi_initialize(self, shape):
        with inf.make_context(use_quasi=True) as context:
            context.set_batch_shape(shape)
            beta = context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))
            context.initialize_parameters()

            assert beta.shape == shape

    @pytest.mark.parametrize("use_quasi", [False, True])
    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_copy_context(self, use_quasi, shape):
        with inf.make_context(use_quasi=use_quasi) as context:
            context.set_batch_shape(shape)
            beta = context.named_parameter("beta", LogNormal(loc=0.0, scale=1.0))
        
        context.initialize_parameters()
        copy_context = context.copy()

        for (ck, cv), (k, v) in zip(copy_context.parameters.items(), context.parameters.items()):
            assert (ck == k) and (cv == v).all() and (cv.prior.equivalent_to(v.prior))

    @pytest.mark.parametrize("use_quasi", [False, True])
    @pytest.mark.parametrize("shape", BATCH_SHAPES)
    def test_threading(self, use_quasi, shape):
        def initialize(uq, s):
            with inf.make_context(uq) as context:
                context.set_batch_shape(s)

                assert (context.stack[-1] is context) and len(context.stack) == 1

        n = 5
        with futures.ThreadPoolExecutor(n) as pool:
            pool.map(initialize, n * ((use_quasi, shape)))
