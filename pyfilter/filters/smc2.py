from .ness import NESS
from ..utils.utils import get_ess
import numpy as np
import scipy.stats as stats
from ..distributions.continuous import Distribution


def _define_pdf(params):
    """
    Helper function for creating the PDF.
    :param params:
    :return:
    """

    mean = params[0].mean()
    std = params[0].std()

    a = (params[1].bounds()[0] - mean) / std
    b = (params[1].bounds()[1] - mean) / std

    return stats.truncnorm(a, b, mean, std)


def _mcmc_move(params):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params:
    :return:
    """

    return _define_pdf(params).rvs(size=params[0].shape)


def _kernel(params, new_params):
    """
    Creates the kernel used for performing the MCMC move.
    :param params:
    :param new_params:
    :return:
    """

    return _define_pdf(params).logpdf(new_params)


class SMC2(NESS):
    def __init__(self, model, particles, threshold=0.2, disp=True, **filtkwargs):

        super().__init__(model, particles, **filtkwargs)

        self._th = threshold
        self._recw = 0      # type: np.ndarray
        self._ior = 0
        self._disp = disp

    def filter(self, y):

        # ===== Perform a filtering move ===== #

        self._filter.filter(y)
        self._recw += self._filter.s_l[-1]

        # ===== Calculate efficient number of samples ===== #

        ess = get_ess(self._recw)

        # ===== Rejuvenate if there are too few samples ===== #

        if ess < self._th * self._particles[0]:
            if self._disp:
                print('\nESS fell below threshold at index {:d} -> rejuvenating'.format(self._ior))
            self._rejuvenate()

        self._ior += 1

        return self

    def _calc_kernels(self, copy):
        """
        Calculates the kernel likelihood of the respective parameters.
        :param copy:
        :return:
        """

        # TODO: Consider doing this somewhere else...

        out = 0
        for nts, ots in zip(copy._model.hidden, self._filter._model.hidden):
            for i in range(len(nts.theta)):
                if isinstance(nts._o_theta[i], Distribution):
                    newkernel = _kernel((nts.theta[i], nts._o_theta[i]), ots.theta[i])
                    oldkernel = _kernel((ots.theta[i], ots._o_theta[i]), nts.theta[i])

                    out += newkernel - oldkernel

        ntso = copy._model.observable
        otso = self._filter._model.observable

        for i in range(len(otso.theta)):
            if isinstance(otso.theta[i], Distribution):
                newkernel = _kernel((ntso.theta[i], ntso._o_theta[i]), otso.theta[i])
                oldkernel = _kernel((otso.theta[i], otso._o_theta[i]), ntso.theta[i])

                out += newkernel - oldkernel

        return out

    def _rejuvenate(self):
        """
        Rejuvenates the particles using a PMCMC move.
        :return:
        """

        # ===== Resample among parameters ===== #

        inds = self._resamp(self._recw)
        self._filter.resample(inds)

        # ===== Define new filters and move via MCMC ===== #

        t_filt = self._filter.copy().reset()
        t_filt._model.p_apply(_mcmc_move)

        # ===== Filter data ===== #

        t_filt.longfilter(self._td[:self._ior+1])

        # ===== Calculate acceptance ratio ===== #

        quotient = np.sum(t_filt.s_l, axis=0) - np.sum(self._filter.s_l, axis=0)
        plogquot = t_filt._model.p_prior() - self._filter._model.p_prior()
        kernel = self._calc_kernels(t_filt)

        # ===== Check which to accept ===== #

        u = np.log(np.random.uniform(size=quotient.size))
        toaccept = u < quotient + plogquot[:, 0] + kernel[:, 0]

        if self._disp:
            print('     Acceptance rate of PMCMC move is {:.1%}'.format(toaccept.mean()))

        # ===== Replace old filters with newly accepted ===== #

        self._filter.exchange(toaccept, t_filt)
        self._recw = 0

        # ===== Increase states if less than 20% are accepted ===== #

        if toaccept.mean() < 0.2:
            if self._disp:
                print('      Acceptance rate fell below threshold - increasing states')
                self._increase_states()

        return self

    def _increase_states(self):
        """
        Increases the number of states.
        :return:
        """

        # ===== Create new filter with double the state particles ===== #

        n_particles = self._filter._particles[0], 2 * self._filter._particles[1]
        t_filt = self._filter.copy().reset(n_particles).longfilter(self._td[:self._ior+1])

        # ===== Calculate new weights and replace filter ===== #

        self._recw = np.sum(t_filt.s_l, axis=0) - np.sum(self._filter.s_l, axis=0)
        self._filter = t_filt

        return self