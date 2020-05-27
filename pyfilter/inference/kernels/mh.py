from ...filters.base import BaseFilter
from ...utils import normalize
from .base import BaseKernel
from ..utils import stacker, _eval_kernel, _construct_mvn, _mcmc_move
import torch
from torch.distributions import MultivariateNormal, Independent, Distribution
from typing import Iterable


class ParticleMetropolisHastings(BaseKernel):
    def __init__(self, nsteps=1, **kwargs):
        """
        Implements a base class for the particle Metropolis Hastings class.
        :param nsteps: The number of steps to perform
        """

        super().__init__(**kwargs)

        self._nsteps = nsteps
        self._y = None
        self.accepted = None
        self._entire_hist = True

    def set_data(self, y: Iterable[torch.Tensor]):
        """
        Sets the data to be used when calculating acceptance probabilities.
        :param y: The data
        :return: Self
        """
        self._y = y

        return self

    def define_pdf(self, values: torch.Tensor, weights: torch.Tensor) -> Distribution:
        """
        The method to be overridden by the user for defining the kernel to propagate the parameters. Note that the
        parameters are propagated in the transformed space.
        :param values: The parameters as a single Tensor
        :param weights: The normalized weights of the particles
        :return: A distribution
        """

        raise NotImplementedError()

    def _calc_diff_logl(self, t_filt: BaseFilter, filter_: BaseFilter):
        """
        Helper method for calculating the difference in log likelihood between proposed and existing parameters.
        :param t_filt: The new filter
        :param filter_: The old filter
        :return: Difference in loglikelihood
        """

        t_filt.reset().initialize().longfilter(self._y, bar=False)
        return t_filt.result.loglikelihood.sum(dim=0) - filter_.result.loglikelihood.sum(dim=0)

    def _before_resampling(self, filter_: BaseFilter, stacked: torch.Tensor):
        """
        Helper method for carrying out operations before resampling.
        :param filter_: The filter
        :param stacked: The stacked parameterss
        :return: Self
        """

        return self

    def _update(self, parameters, filter_, weights):
        for i in range(self._nsteps):
            # ===== Construct distribution ===== #
            stacked = stacker(parameters, lambda u: u.t_values)
            dist = self.define_pdf(stacked.concated, weights)

            # ===== Perform necessary operation prior to resampling ===== #
            self._before_resampling(filter_, stacked.concated)

            # ===== Resample among parameters ===== #
            inds = self._resampler(weights, normalized=True)
            filter_.resample(inds, entire_history=self._entire_hist)

            # ===== Define new filters and move via MCMC ===== #
            t_filt = filter_.copy()
            t_filt.viewify_params((*filter_._n_parallel, 1))
            _mcmc_move(t_filt.ssm.theta_dists, dist, stacked, stacked.concated.shape[0])

            # ===== Calculate difference in loglikelihood ===== #
            quotient = self._calc_diff_logl(t_filt, filter_)

            # ===== Calculate acceptance ratio ===== #
            plogquot = t_filt.ssm.p_prior() - filter_.ssm.p_prior()
            kernel = _eval_kernel(filter_.ssm.theta_dists, dist, t_filt.ssm.theta_dists)

            # ===== Check which to accept ===== #
            u = torch.empty_like(quotient).uniform_().log()
            toaccept = u < quotient + plogquot + kernel

            # ===== Update the description ===== #
            self.accepted = toaccept.sum().float() / float(toaccept.shape[0])

            if self._entire_hist:
                filter_.exchange(t_filt, toaccept)
            else:
                filter_.ssm.exchange(toaccept, t_filt.ssm)

            weights = normalize(filter_.result.loglikelihood.sum(dim=0))

        return self


class SymmetricMH(ParticleMetropolisHastings):
    def define_pdf(self, values, weights):
        return _construct_mvn(values, weights)
