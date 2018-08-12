from .ness import NESS
from ..utils.utils import get_ess
import numpy as np
import scipy.stats as stats
from ..distributions.continuous import Distribution


def _define_pdf(params):
    """
    Helper function for creating the PDF.
    :param params: The parameters to use for defining the distribution
    :type params: Distribution
    :return: A truncated normal distribution
    :rtype: stats.truncnorm
    """

    values = params.values

    mean = values.mean()
    std = values.std()

    low, high = params.bounds()
    a, b = (low - mean) / std, (high - mean) / std

    return stats.truncnorm(a=a, b=b, loc=mean, scale=std)


def _mcmc_move(params):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params: The parameters to use for defining the distribution
    :type params: Distribution
    :return: Samples from a truncated normal distribution
    :rtype: np.ndarray
    """

    return _define_pdf(params).rvs(size=params.values.shape)


def _eval_kernel(params, new_params):
    """
    Evaluates the kernel used for performing the MCMC move.
    :param params: The current parameters
    :type params: Distribution
    :param new_params: The new parameters to evaluate against
    :type new_params: Distribution
    :return: The log density of the proposal kernel evaluated at `new_params`
    :rtype: np.ndarray
    """

    return _define_pdf(params).logpdf(new_params.values)


class SMC2(NESS):
    def __init__(self, model, particles, threshold=0.2, disp=True, **filtkwargs):
        """
        Implements the SMC2 algorithm by Chopin et al.
        :param model: See BaseFilter
        :param particles: See BaseFilter
        :param threshold: The threshold at which to perform MCMC rejuvenation
        :type threshold: float
        :param disp: Whether or not to display when the algorithm performs a rejuvenation step
        :type disp: bool
        :param filtkwargs: kwargs passed to the filter targeting the states
        """
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
        :param copy: The copy of the filter
        :type copy: pyfilter.filters.base.BaseFilter
        :return:
        """

        processes = [
            (copy.ssm.hidden.theta_dists, self.ssm.hidden.theta_dists),
            (copy.ssm.observable.theta_dists, self.ssm.observable.theta_dists)

        ]

        out = 0
        for proc in processes:
            for newp, oldp in zip(*proc):
                out += _eval_kernel(newp, oldp) - _eval_kernel(oldp, newp)

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
        t_filt._model.p_apply(_mcmc_move, transformed=False)

        # ===== Filter data ===== #

        t_filt.longfilter(self._td[:self._ior+1])

        # ===== Calculate acceptance ratio ===== #

        quotient = np.sum(t_filt.s_l, axis=0) - np.sum(self._filter.s_l, axis=0)
        plogquot = t_filt._model.p_prior() - self._filter._model.p_prior()
        kernel = self._calc_kernels(t_filt)

        # ===== Check which to accept ===== #

        u = np.log(np.random.uniform(size=quotient.shape))
        if plogquot.ndim > 1:
            toaccept = u < quotient + plogquot[:, 0] + kernel[:, 0]
        else:
            toaccept = u < quotient + plogquot + kernel

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
        # TODO: Something goes wrong here
        n_particles = self._filter._particles[0], 2 * self._filter._particles[1]
        t_filt = self._filter.copy().reset(n_particles).longfilter(self._td[:self._ior+1])

        # ===== Calculate new weights and replace filter ===== #

        self._recw = np.sum(t_filt.s_l, axis=0) - np.sum(self._filter.s_l, axis=0)
        self._filter = t_filt

        return self