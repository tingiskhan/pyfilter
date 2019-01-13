from .base import Proposal
from ..unscentedtransform import UnscentedTransform
from ..utils import choose


class Unscented(Proposal):
    """
    Implements the unscented proposal by van der Merwe et al.
    """

    def set_model(self, model):
        self._model = model
        self._kernel = UnscentedTransform(self._model)

        return self

    def draw(self, y, x):
        if not self._kernel.initialized:
            self._kernel.initialize(x)

        self._kernel = self._kernel.construct(y)

        return self._kernel.x_dist.sample()

    def weight(self, y, xn, xo, *args, **kwargs):
        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - self._kernel.x_dist.log_prob(xn)

    def resample(self, inds):
        if not self._kernel.initialized:
            return self

        self._kernel.xmean = choose(self._kernel.xmean, inds)
        self._kernel.xcov = choose(self._kernel.xcov, inds)

        return self
