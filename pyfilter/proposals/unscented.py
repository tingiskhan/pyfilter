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
        if not (isinstance(model.observable, AffineProcess) and isinstance(model.hidden, AffineProcess)):
            raise ValueError(f'Both observable and hidden must be of type {AffineProcess.__class__.__name__}!')

        self._model = model
        self._ut = UnscentedTransform(self._model)

        return self

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
