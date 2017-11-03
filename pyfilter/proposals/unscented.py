from ..proposals import Linearized
import numpy as np
from ..utils.utils import choose, customcholesky
from ..utils.unscentedtransform import UnscentedTransform
from ..distributions.continuous import MultivariateNormal, Normal


class Unscented(Linearized):
    def __init__(self, model, *args):
        super().__init__(model, *args)
        self.ut = UnscentedTransform(model)     # type: UnscentedTransform

    def draw(self, y, x, size=None, *args, **kwargs):
        mean, cov = self.ut.construct(y)

        if self._model.hidden_ndim > 1:
            self._kernel = MultivariateNormal(mean, customcholesky(cov))
        else:
            self._kernel = Normal(mean[0], np.sqrt(cov[0, 0]))

        return self._kernel.rvs(size=size)

    def resample(self, inds):
        self.ut._mean = choose(self.ut._mean, inds)
        self.ut._cov = choose(self.ut._cov, inds)

        return self