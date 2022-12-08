from pyro.distributions import Distribution


class TestPrior(object):
    # TODO: Should verify that all classes have...
    def test_verify_is_patched(self):
        from pyfilter.inference import PriorMixin

        for method_name in (mn for mn in dir(PriorMixin) if callable(mn)):
            assert hasattr(Distribution, method_name)
