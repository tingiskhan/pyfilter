import itertools

import pytest
from pyfilter import inference as inf, filters as filts
from models import linear_models


def algorithms():
    yield lambda f: inf.sequential.NESS(f, 2_000)
    yield lambda f: inf.sequential.SMC2(f, 2_000, num_steps=5)
    yield lambda f: inf.sequential.SMC2(f, 2_000, num_steps=10, distance_threshold=0.1)


def callbacks():
    yield inf.sequential.collectors.MeanCollector()
    yield lambda *u: None


PARAMETERS = itertools.product(linear_models(), algorithms(), callbacks())


class TestSequential(object):
    @pytest.mark.parametrize("models, algorithm, callback", PARAMETERS)
    def test_algorithms(self, models, algorithm, callback):
        true_model, build_model = models
        _, y = true_model.sample_states(1_000).get_paths()

        with inf.make_context() as context:
            filter_ = filts.APF(build_model, 200)
            alg = algorithm(filter_)

            alg.register_callback(callback)

            result = alg.fit(y)

            # TODO: Add something to test
            print()
