from pyfilter.inference import Prior
from pyro.distributions import Normal


class TestPrior(object):
    def test_copy_prior(self):
        prior = Prior(Normal, loc=0.0, scale=1.0)

        copied_prior = prior.copy()

        assert (copied_prior is not prior) and (prior.equivalent_to(copied_prior))
