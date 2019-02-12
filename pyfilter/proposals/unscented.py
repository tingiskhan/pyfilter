from .base import Proposal
from ..unscentedtransform import UnscentedTransform
from ..utils import choose


class Unscented(Proposal):
    def __init__(self):
        """
        Implements the unscented proposal by van der Merwe et al.
        """
        super().__init__()
        self._ut = None

    def set_model(self, model):
        self._model = model
        self._ut = UnscentedTransform(self._model)

        return self

    def construct(self, y, x):
        if not self._ut.initialized:
            self._ut.initialize(x)

        self._ut = self._ut.construct(y)
        self._kernel = self._ut.x_dist

        return self

    def resample(self, inds):
        if not self._ut.initialized:
            return self

        self._ut.xmean = choose(self._ut.xmean, inds)
        self._ut.xcov = choose(self._ut.xcov, inds)

        return self
