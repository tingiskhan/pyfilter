from .base import Proposal
from ..unscentedtransform import UnscentedTransform
from ..utils import choose
from ..timeseries import AffineProcess


class Unscented(Proposal):
    def __init__(self):
        """
        Implements the unscented proposal by van der Merwe et al.
        """
        super().__init__()
        self._ut = None
        self._initialized = False

    def set_model(self, model):
        self._model = model
        self._ut = UnscentedTransform(self._model)

        return self

    def modules(self):
        return {'_ut': self._ut} if self._ut is not None else {}

    def construct(self, y, x):
        if not self._initialized:
            self._ut.initialize(x)

        self._ut = self._ut.construct(y)
        self._kernel = self._ut.x_dist

        return self

    def resample(self, inds):
        if not self._initialized:
            return self

        self._ut.xmean = choose(self._ut.xmean, inds)
        self._ut.xcov = choose(self._ut.xcov, inds)

        return self
