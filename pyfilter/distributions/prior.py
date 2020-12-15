from .parameterized_distribution import DistributionWrapper


class Prior(DistributionWrapper):
    def __init__(self, base_dist, **parameters):
        super().__init__(base_dist, **parameters)

        if any(self._parameters):
            raise NotImplementedError("Priors currently do not support parameters!")
