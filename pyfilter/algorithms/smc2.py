from .ness import NESS
from pyfilter.utils.utils import get_ess, expanddims, normalize
import numpy as np
from pyfilter.distributions.continuous import Distribution, MultivariateNormal
from pyfilter.filters.base import KalmanFilter


def _define_pdf(params, weights):
    """
    Helper function for creating the PDF.
    :param params: The parameters to use for defining the distribution
    :type params: tuple of tuple of Distribution
    :param weights: The weights to use
    :type weights: np.ndarray
    :return: A truncated normal distribution
    :rtype: stats.truncnorm
    """

    asarray = np.array([p.t_values for p in params])

    if asarray.ndim > 2:
        asarray = asarray[..., 0]

    mean = (asarray * weights).sum(axis=-1)
    centralized = asarray.T - mean
    cov = np.einsum('ij,jk->ik', weights * centralized.T, centralized)

    return MultivariateNormal(mean, np.linalg.cholesky(cov))


def _mcmc_move(params, dist):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params: The parameters to use for defining the distribution
    :type params: tuple of Distribution
    :param dist: The distribution to use for sampling
    :type dist: stats.multivariate_normal
    :return: Samples from a truncated normal distribution
    :rtype: np.ndarray
    """
    shape = next(p.t_values.shape for p in params)
    if len(shape) > 1:
        shape = shape[:-1]

    rvs = dist.rvs(size=shape)

    for p, vals in zip(params, rvs):
        p.t_values = expanddims(vals, p.t_values.ndim)

    return True


def _eval_kernel(params, dist, n_params, n_dist):
    """
    Evaluates the kernel used for performing the MCMC move.
    :param params: The current parameters
    :type params: tuple of Distribution
    :param n_params: The new parameters to evaluate against
    :type n_params: tuple of Distribution
    :return: The log density of the proposal kernel evaluated at `new_params`
    :rtype: np.ndarray
    """

    p_vals = np.array([p.t_values for p in params])
    n_p_vals = np.array([p.t_values for p in n_params])

    return n_dist.logpdf(p_vals) - dist.logpdf(n_p_vals)


class SMC2(NESS):
    def __init__(self, model, particles, threshold=0.2, disp=False, **filtkwargs):
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

    def _rejuvenate(self):
        """
        Rejuvenates the particles using a PMCMC move.
        :return:
        """

        # ===== Construct distribution ===== #
        ll = np.sum(self._filter.s_l, axis=0)
        dist = _define_pdf(self._filter.ssm.flat_theta_dists, normalize(self._recw))

        # ===== Resample among parameters ===== #

        inds = self._resamp(self._recw)
        self._filter.resample(inds)

        # ===== Define new filters and move via MCMC ===== #

        t_filt = self._filter.copy().reset()
        _mcmc_move(t_filt.ssm.flat_theta_dists, dist)

        # ===== Filter data ===== #

        t_filt.longfilter(self._td[:self._ior+1], bar=False)
        t_ll = np.sum(t_filt.s_l, axis=0)

        # ===== Calculate acceptance ratio ===== #
        # TODO: Might have to add gradients for transformation?
        quotient = t_ll - ll[inds]
        plogquot = t_filt._model.p_prior() - self._filter._model.p_prior()
        kernel = _eval_kernel(self._filter.ssm.flat_theta_dists, dist, t_filt.ssm.flat_theta_dists, dist)

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
        self._recw = np.zeros_like(self._recw)

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

        if isinstance(self._filter, KalmanFilter):
            return self

        # ===== Create new filter with double the state particles ===== #
        # TODO: Something goes wrong here
        n_particles = self._filter._particles[0], 2 * self._filter._particles[1]
        t_filt = self._filter.copy().reset(n_particles).longfilter(self._td[:self._ior+1], bar=False)

        # ===== Calculate new weights and replace filter ===== #

        self._recw = np.sum(t_filt.s_l, axis=0) - np.sum(self._filter.s_l, axis=0)
        self._filter = t_filt

        return self