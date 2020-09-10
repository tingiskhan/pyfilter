from .base import Proposal
from ..uft import UnscentedFilterTransform, UFTCorrectionResult
from ..utils import choose


class Unscented(Proposal):
    def __init__(self):
        """
        Implements the unscented proposal by van der Merwe et al.
        """
        super().__init__()
        self._ut = None         # type: UnscentedFilterTransform
        self._ut_res = None     # type: UFTCorrectionResult

    def set_model(self, model):
        self._model = model
        self._ut = UnscentedFilterTransform(self._model)

        return self

    def construct(self, y, x):
        if self._ut_res is None:
            self._ut_res = self._ut.initialize(x.shape[:-1] if self._model.hidden_ndim > 0 else x.shape)

        p = self._ut.predict(self._ut_res)
        self._ut_res = self._ut.correct(y, p)
        self._kernel = self._ut_res.x_dist()

        return self

    def resample(self, inds):
        if self._ut_res is None:
            return self

        self._ut_res.xm = choose(self._ut_res.xm, inds)
        self._ut_res.xc = choose(self._ut_res.xc, inds)

        return self

    def pre_weight(self, y, x):
        p = self._ut.predict(self._ut_res)
        m = self._ut.calc_mean_cov(p)

        logprob = self._model.log_prob(y, m.xm)

        if self._model.hidden_ndim < 1:
            return logprob[..., 0]

        return logprob