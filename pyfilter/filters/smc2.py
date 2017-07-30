from .ness import NESS
from ..helpers.helpers import get_ess
from ..helpers.resampling import systematic
import numpy as np
import scipy.stats as stats


def _mcmc_move(params):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params:
    :return:
    """

    mean = params[0].mean()
    std = params[0].std()

    a = (params[1].bounds()[0] - mean) / std
    b = (params[1].bounds()[1] - mean) / std

    return stats.truncnorm.rvs(a, b, mean, std, size=params[0].shape)


class SMC2(NESS):
    def __init__(self, model, particles, *args, threshold=0.2, disp=True, **kwargs):

        super().__init__(model, particles, *args, **kwargs)

        self._th = threshold
        self._recw = 0      # type: np.ndarray
        self._ior = 0
        self._disp = disp

    def filter(self, y):

        self._filter.filter(y)

        self._recw += self._filter.s_l[-1]

        ess = get_ess(self._recw)

        if ess < 0.2 * self._particles[0]:
            if self._disp:
                print('ESS fell below threshold at index {:d} -> rejuvenating'.format(self._ior))
            self._rejuvenate()

        self._ior += 1

    def _rejuvenate(self):
        inds = systematic(self._recw)

        self._filter.resample(inds)

        copy = self._filter.copy().reset()
        copy._model.p_apply(_mcmc_move)

        copy.longfilter(self._td[:self._ior+1])

        quotient = np.sum(copy.s_l, axis=0) - np.sum(self._filter.s_l, axis=0)
        plogquot = copy._model.p_prior() - self._filter._model.p_prior()

        u = np.log(np.random.uniform(size=quotient.size))

        toaccept = u < quotient + plogquot[:, 0]

        if self._disp:
            print(' Acceptance rate of PMCMC move is {:.1%}'.format(toaccept.mean()))

        self._filter.replace(toaccept, copy)

        self._recw = 0

        return self