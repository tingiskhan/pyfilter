import pytest
import torch
from functools import partial

from pyfilter import inference as inf, filters as filts
from .models import linear_models


def algorithms(particles=400):
    yield partial(inf.sequential.NESS, particles=particles)
    yield partial(inf.sequential.SMC2, particles=particles, num_steps=5)
    yield partial(inf.sequential.SMC2, particles=particles, num_steps=10, distance_threshold=0.1)
    yield partial(inf.sequential.NESSMC2, particles=particles)


def callbacks():
    yield inf.sequential.collectors.MeanCollector()
    yield None


def quasi_supported_algorithms(particles=400):
    yield lambda f: inf.sequential.SMC2(f, particles)


class TestSequential(object):
    @classmethod
    def do_test_fit(cls, true_model, build_model, algorithm, callback, **kwargs):
        torch.manual_seed(123)
        _, y = true_model.sample_states(100).get_paths()

        with inf.make_context(**kwargs) as context:
            filter_ = filts.particle.APF(build_model, 200, proposal=filts.particle.proposals.LinearGaussianObservations())
            alg = algorithm(filter_)

            alg.register_callback(callback)

            # TODO: Add something to test
            result = alg.fit(y)

    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("algorithm", algorithms())
    @pytest.mark.parametrize("callback", callbacks())
    def test_algorithms(self, models, algorithm, callback):
        self.do_test_fit(*models, algorithm, callback)

    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("algorithm", algorithms())
    @pytest.mark.parametrize("callback", callbacks())
    def test_algorithms_serialize(self, models, algorithm, callback):
        torch.manual_seed(123)

        true_model, build_model = models
        _, y = true_model.sample_states(100).get_paths()

        train_split = y.shape[0] // 2
        particles = 250
        with inf.make_context() as context:
            filter_ = filts.APF(build_model, particles)
            alg = algorithm(filter_)

            alg.register_callback(callback)

            result = alg.fit(y[:train_split])

            algorithm_state = result.state_dict()
            context_state = context.state_dict()

        with inf.make_context() as new_context:
            new_filter = filts.APF(build_model, filter_.particles[-1])
            new_alg = algorithm(new_filter)
            new_result = new_alg.initialize()

            new_context.load_state_dict(context_state)                        
            new_result.load_state_dict(algorithm_state)

            assert (
                (new_result.ess == result.ess).all() and
                (new_result.w == result.w).all()
            )

            for yt in y[train_split:]:
                new_result = new_alg.step(yt, new_result)

            assert (
                    (new_result.ess.shape[0] == y.shape[0] + 1) and
                    (new_result.filter_state.latest_state.timeseries_state.time_index == y.shape[0]).all()
            )

    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("algorithm", quasi_supported_algorithms())
    @pytest.mark.parametrize("randomize", [False, True])
    def test_quasi(self, models, algorithm, randomize):
        self.do_test_fit(*models, algorithm, None, use_quasi=True, randomize=randomize)
