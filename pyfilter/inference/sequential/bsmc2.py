from .smc2 import SMC2
from .base import SequentialParticleAlgorithm
from ..kernels.bmh import BlockMetropolisHastings
from ..utils import _construct_mvn
from ...utils import normalize


class BSMC2(SequentialParticleAlgorithm):
    def __init__(self, filter_, particles, switch=500, block_length=250, **kwargs):
        """
        A blocked version of SMC2, inspired by .. . But instead of being online, this algorithm is only sequential. Is
        less costly than SMC2 for longer timeseries.
        Do note that this is one of my creations, as such, use with care.
        :param switch: When to switch to using blocks of data instead of entire history.
        :param block_length: The minimum size of one block to consider.
        """

        super().__init__(filter_, particles)
        self._switch = switch
        self._bl = block_length
        self._num_iters = 0

        self._kw = kwargs
        self._alg = SMC2(self.filter, particles, threshold=kwargs.pop("threshold", 0.5), **kwargs)

    def _update(self, y, state):
        self._num_iters += 1

        if (self._num_iters % self._bl == 0) and (self._num_iters >= self._switch):
            state = self._alg.rejuvenate(state)

            mvn = _construct_mvn(self.filter.ssm.parameters_as_matrix().concated, normalize(state.w))
            bmh = BlockMetropolisHastings(state.filter_state, mvn)

            self._alg = SMC2(self.filter.reset(only_ll=True), self.particles[0], kernel=bmh, threshold=0., **self._kw)

        return self._alg.update(y, state)

    def predict(self, steps, state, aggregate=True, **kwargs):
        return self._alg.predict(steps, state, aggregate, **kwargs)



