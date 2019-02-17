from .ness import NESS
import torch
from ..utils import get_ess, add_dimensions, normalize
from ..filters.base import KalmanFilter, ParticleFilter
from torch.distributions import MultivariateNormal
from time import sleep
from ..resampling import systematic
from sklearn.gaussian_process import GaussianProcessRegressor
from numpy import float32


def _define_pdf(params, weights):
    """
    Helper function for creating the PDF.
    :param params: The parameters to use for defining the distribution
    :type params: tuple[Distribution]
    :param weights: The weights to use
    :type weights: torch.Tensor
    :return: A multivariate normal distribution
    :rtype: MultivariateNormal
    """

    asarray = torch.cat([p.t_values for p in params], dim=-1)

    if asarray.dim() > 2:
        asarray = asarray[..., 0]

    mean = (asarray * weights[:, None]).sum(0)
    centralized = asarray - mean
    cov = torch.matmul(weights * centralized.t(), centralized)

    return MultivariateNormal(mean, scale_tril=torch.cholesky(cov))


def _mcmc_move(params, dist):
    """
    Performs an MCMC move to rejuvenate parameters.
    :param params: The parameters to use for defining the distribution
    :type params: tuple[Distribution]
    :param dist: The distribution to use for sampling
    :type dist: MultivariateNormal
    :return: Samples from a multivariate normal distribution
    :rtype: torch.Tensor
    """
    shape = next(p.t_values.shape for p in params)
    if len(shape) > 1:
        shape = shape[:-1]

    rvs = dist.sample(shape)

    for p, vals in zip(params, rvs.t()):
        p.t_values = add_dimensions(vals, p.t_values.dim())

    return True


def _eval_kernel(params, dist, n_params):
    """
    Evaluates the kernel used for performing the MCMC move.
    :param params: The current parameters
    :type params: tuple[Distribution]
    :param dist: The distribution to use for evaluating the prior
    :type dist: MultivariateNormal
    :param n_params: The new parameters to evaluate against
    :type n_params: tuple of Distribution
    :return: The log difference in priors
    :rtype: torch.Tensor
    """

    p_vals = torch.cat([p.t_values for p in params], dim=-1)
    n_p_vals = torch.cat([p.t_values for p in n_params], dim=-1)

    return dist.log_prob(p_vals) - dist.log_prob(n_p_vals)


class SMC2(NESS):
    def __init__(self, filter_, particles, threshold=0.2, resampling=systematic):
        """
        Implements the SMC2 algorithm by Chopin et al.
        :param particles: The amount of particles
        :type particles: int
        :param threshold: The threshold at which to perform MCMC rejuvenation
        :type threshold: float
        """

        if isinstance(filter_, KalmanFilter):
            raise ValueError('`filter_` must be of instance `{:s}!'.format(ParticleFilter.__name__))

        super().__init__(filter_, particles, resampling=resampling)

        self._th = threshold
        self._switch = float("inf")

        self._gpr = None    # type: GaussianProcessRegressor
        self._lastrejuv = 0

    def _update(self, y):
        # ===== Perform a filtering move ===== #
        self._y += (y,)
        self.filter.filter(y)
        self._w_rec += self.filter.s_ll[-1]

        # ===== Calculate efficient number of samples ===== #
        ess = get_ess(self._w_rec)
        self._logged_ess += (ess,)

        # ===== Rejuvenate if there are too few samples ===== #
        if ess < self._th * self._w_rec.shape[0]:
            self._rejuvenate()
            self._iterator.set_description(desc=str(self))

            self._lastrejuv = len(self._y)

        return self

    def _rejuvenate(self):
        """
        Rejuvenates the particles using a PMCMC move.
        :return: Self
        :rtype: SMC2
        """

        # ===== Update the description ===== #
        self._iterator.set_description(desc='{:s} - Rejuvenating particles'.format(str(self)))

        # ===== Construct distribution ===== #
        dist = _define_pdf(self.filter.ssm.flat_theta_dists, normalize(self._w_rec))

        # ===== Resample among parameters ===== #
        inds = self._resampler(self._w_rec)
        self.filter.resample(inds, entire_history=True)

        # ===== Define new filters and move via MCMC ===== #
        t_filt = self.filter.copy()
        _mcmc_move(t_filt.ssm.flat_theta_dists, dist)

        # ===== Filter data ===== #
        if len(self._y) < self._switch:
            t_filt.reset().initialize().longfilter(self._y, bar=False)

            quotient = t_filt.loglikelihood - self.filter.loglikelihood
        else:
            params = torch.cat([p.t_values for p in self.filter.ssm.flat_theta_dists], dim=-1)

            filt_logl = sum(self.filter.s_ll[-self._lastrejuv:])
            gpr = self._gpr.fit(params.cpu(), filt_logl.reshape(-1, 1).cpu())

            # TODO: How to handle nans?
            # TODO: Use gpytorch

            pred_logl = torch.tensor(
                gpr.predict(torch.cat([p.t_values for p in t_filt.ssm.flat_theta_dists], dim=-1).cpu()).astype(float32)
            )[:, 0]

            pred_logl[torch.isnan(pred_logl)] = float("-inf")
            quotient = pred_logl - filt_logl

        # ===== Calculate acceptance ratio ===== #
        plogquot = t_filt.ssm.p_prior() - self.filter.ssm.p_prior()
        kernel = _eval_kernel(self.filter.ssm.flat_theta_dists, dist, t_filt.ssm.flat_theta_dists)

        # ===== Check which to accept ===== #
        u = torch.empty(quotient.shape).uniform_().log()
        if plogquot.dim() > 1:
            toaccept = u < quotient + plogquot[:, 0] + kernel
        else:
            toaccept = u < quotient + plogquot + kernel

        # ===== Update the description ===== #
        accepted = float(toaccept.sum()) / float(toaccept.shape[0])
        self._iterator.set_description(desc='{:s} - Accepted particles is {:.1%}'.format(str(self), accepted))
        sleep(1)

        # ===== Replace old filters with newly accepted ===== #
        self.filter.exchange(t_filt, toaccept)
        self._w_rec *= 0.

        # ===== Increase states if less than 20% are accepted ===== #
        if accepted < 0.2:
            self._increase_states()

        return self

    def _increase_states(self):
        """
        Increases the number of states.
        :return: Self
        :rtype: SMC2
        """

        # ===== Create new filter with double the state particles ===== #
        t_filt = self.filter.copy().reset()
        t_filt._particles = 2 * self.filter._particles[1]

        msg = '{:s} - Increasing number of state particles from {:d} -> {:d}'
        self._iterator.set_description(desc=msg.format(str(self), self._filter.particles[-1], t_filt._particles))

        t_filt.set_nparallel(self._w_rec.shape[0]).initialize().longfilter(self._y, bar=False)

        # ===== Calculate new weights and replace filter ===== #
        self._w_rec = t_filt.loglikelihood - self.filter.loglikelihood
        self.filter = t_filt

        return self


class GPSMC2(SMC2):
    def __init__(self, filter_, particles, switch=250, gpr=GaussianProcessRegressor(normalize_y=True), **kwargs):
        """
        Implements an algorithm similar to that of ...
        :param switch: The point at which to switch to "dynamic" proposals
        :type switch: int
        """
        super().__init__(filter_, particles, **kwargs)
        self._window = switch
        self._gpr = gpr
