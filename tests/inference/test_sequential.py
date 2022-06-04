import itertools

import pytest
from pyfilter import inference as inf, filters as filts
from .models import linear_models, build_model


def algorithms():
    yield lambda f: inf.sequential.NESS(f, 1_000)
    yield lambda f: inf.sequential.SMC2(f, 1_000, num_steps=5)


PARAMETERS = itertools.product(linear_models(), algorithms())


class TestSequential(object):
    @pytest.mark.parametrize("true_model, algorithm", PARAMETERS)
    def test_smc2(self, true_model, algorithm):
        _, y = true_model.sample_states(500).get_paths()

        with inf.make_context() as context:
            filter_ = filts.APF(build_model, 250)
            alg = algorithm(filter_)

            result = alg.fit(y)

            # TODO: Add something to test
            print()

