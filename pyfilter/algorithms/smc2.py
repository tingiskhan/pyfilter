from .ness import NESS
import torch
from ..utils import get_ess, add_dimensions, normalize
from ..filters.base import KalmanFilter
from torch.distributions import MultivariateNormal


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

    asarray = torch.cat([p.t_values for p in params], 1)

    if asarray.dim() > 2:
        asarray = asarray[..., 0]

    mean = (asarray * weights[:, None]).sum(0)
    centralized = asarray - mean
    cov = torch.einsum('ij,jk->ik', (weights * centralized.t(), centralized))

    return MultivariateNormal(mean, cov)


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

    rvs = dist.sample(shape)

    for p, vals in zip(params, rvs.t()):
        p.t_values = add_dimensions(vals, p.t_values.dim())

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

    p_vals = torch.cat([p.t_values for p in params], 1)
    n_p_vals = torch.cat([p.t_values for p in n_params], 1)

    return n_dist.log_prob(p_vals) - dist.log_prob(n_p_vals)


class SMC2(NESS):
    def __init__(self, filter_, particles, threshold=0.2):
        """
        Implements the SMC2 algorithm by Chopin et al.
        :param particles: The amount of particles
        :type particles: int
        :param threshold: The threshold at which to perform MCMC rejuvenation
        :type threshold: float
        """
        super().__init__(filter_, particles)

        self._th = threshold

    def update(self, y):
        # ===== Perform a filtering move ===== #
        self._y += (y,)
        self.filter.filter(y)
        self._w_rec += self.filter.s_ll[-1]

        # ===== Calculate efficient number of samples ===== #
        ess = get_ess(self._w_rec)

        # ===== Rejuvenate if there are too few samples ===== #
        if ess < self._th * self._w_rec.shape[0]:
            self._rejuvenate()

        return self

    def _rejuvenate(self):
        """
        Rejuvenates the particles using a PMCMC move.
        :return:
        """

        # ===== Construct distribution ===== #
        ll = self.filter.loglikelihood
        dist = _define_pdf(self.filter.ssm.flat_theta_dists, normalize(self._w_rec))

        # ===== Resample among parameters ===== #
        inds = self._resampler(self._w_rec)
        self.filter.resample(inds, entire_history=True)

        # ===== Define new filters and move via MCMC ===== #
        t_filt = self.filter.copy().reset().initialize()
        _mcmc_move(t_filt.ssm.flat_theta_dists, dist)

        # ===== Filter data ===== #
        t_filt.longfilter(self._y, bar=False)
        t_ll = t_filt.loglikelihood

        # ===== Calculate acceptance ratio ===== #
        # TODO: Might have to add gradients for transformation?
        quotient = t_ll - ll[inds]
        plogquot = t_filt.ssm.p_prior() - self.filter.ssm.p_prior()
        kernel = _eval_kernel(self.filter.ssm.flat_theta_dists, dist, t_filt.ssm.flat_theta_dists, dist)

        # ===== Check which to accept ===== #

        u = torch.empty(quotient.shape).uniform_().log()
        if plogquot.dim() > 1:
            toaccept = u < quotient + plogquot[:, 0] + kernel
        else:
            toaccept = u < quotient + plogquot + kernel

        # ===== Replace old filters with newly accepted ===== #
        self.filter.exchange(t_filt, toaccept)
        self._w_rec *= 0.

        # ===== Increase states if less than 20% are accepted ===== #
        if toaccept.sum() < 0.2 * toaccept.shape[0]:
            self._increase_states()

        return self

    def _increase_states(self):
        """
        Increases the number of states.
        :return:
        """

        if isinstance(self.filter, KalmanFilter):
            return self

        # ===== Create new filter with double the state particles ===== #
        t_filt = self.filter.copy().reset()
        t_filt._particles = 2 * self.filter._particles[1]
        t_filt.set_nparallel(self._w_rec.shape[0]).initialize().longfilter(self._y, bar=False)

        # ===== Calculate new weights and replace filter ===== #
        self._w_rec = t_filt.loglikelihood - self.filter.loglikelihood
        self.filter = t_filt

        return self