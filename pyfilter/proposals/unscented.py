from .base import Proposal
from ..unscentedtransform import UnscentedTransform


class Unscented(Proposal):
    _ut = None  # type: UnscentedTransform

    def set_model(self, model):
        self._model = model
        self._ut = UnscentedTransform(self._model)

        return self

    def draw(self, y, x, size=None, *args, **kwargs):
        if not self._ut.initialized:
            self._ut.initialize(x)

        self._ut = self._ut.construct(y)

        return self._ut.x_dist.sample()

    def weight(self, y, xn, xo, *args, **kwargs):
        return self._model.weight(y, xn) + self._model.h_weight(xn, xo) - self._ut.x_dist.log_prob(xn)