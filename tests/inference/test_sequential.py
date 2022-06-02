import pytest
from pyfilter import inference as inf, filters as filts
from .models import linear_models, build_model


PARAMETERS = linear_models()


class TestPMCMC(object):
    @pytest.mark.parametrize("true_model", PARAMETERS)
    def test_smc2(self, true_model):
        _, y = true_model.sample_states(500).get_paths()

        with inf.make_context() as context:
            filter_ = filts.APF(build_model, 250)
            smc2 = inf.sequential.SMC2(filter_, 1_000, num_steps=5)

            result = smc2.fit(y)

            # TODO: Add something to test
            print()