from .joint import JointStochasticProcess
from ..distributions import JointDistribution


class ChainedStochasticProcess(JointStochasticProcess):
    """
    Implements a stochastic process which constitutes multiple "chained" stochastic processes, s.t. each process is
    conditionally independent given the previous process.

    # TODO: math
    """

    def build_density(self, x):
        return JointDistribution(
            *(self._modules[name].build_density(x[: i + 1]) for i, name in enumerate(self._proc_names)),
            indices=self.indices
        )
