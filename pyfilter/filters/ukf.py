from .base import BaseFilter
from ..utils.unscentedtransform import UnscentedTransform
from ..distributions.continuous import Normal, MultivariateNormal
import numpy as np


class UKF(BaseFilter):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, particles=0, *args, **kwargs)

        self._ut = UnscentedTransform(model)

    def initialize(self):
        self._initialize_parameters()
        self._ut.initialize(self._model.hidden.i_mean())

        return self

    def filter(self, y):
        self._ut.construct(y)

        if self._model.obs_ndim < 2:
            kernel = Normal(self._ut.ymean[0], np.sqrt(self._ut.ycov[0, 0]))
        else:
            kernel = MultivariateNormal(self._ut.ymean, np.linalg.cholesky(self._ut.ycov))

        self.s_mx.append(self._ut.xmean.copy())
        self.s_l.append(kernel.logpdf(y))

        return self