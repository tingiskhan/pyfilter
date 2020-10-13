from ...utils import unflattify
from .base import BaseKernel
from ..utils import _construct_mvn
import torch
from torch.distributions import Distribution, MultivariateNormal, Independent
from typing import Iterable, Tuple
from math import sqrt
from ...filters import BaseState, BaseFilter, FilterResult


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

    def set_data(self, y: Iterable[torch.Tensor]):
        """
        Sets the data to be used when calculating acceptance probabilities.
        :param y: The data
        :return: Self
        """
        self._y = y

        return self

    def define_pdf(self, values: torch.Tensor, weights: torch.Tensor, inds: torch.Tensor) -> Distribution:
        """
        The method to be overridden by the user for defining the kernel to propagate the parameters. Note that the
        parameters are propagated in the transformed space.
        :param values: The parameters as a single Tensor
        :param weights: The normalized weights of the particles
        :param inds: The resampled indices
        :return: A distribution
        """

        raise NotImplementedError()

    def calc_model_loss(self, new_filter: BaseFilter, old_filter: BaseFilter,
                        old_res: FilterResult) -> Tuple[FilterResult, torch.Tensor]:
        new_res = new_filter.longfilter(self._y, bar=False)
        diff_logl = new_res.loglikelihood - old_res.loglikelihood

        diff_prior = new_filter.ssm.p_prior() - old_filter.ssm.p_prior()

        return new_res, diff_logl + diff_prior

    def _update(self, parameters, filter_, state, weights):
        for i in range(self._nsteps):
            # ===== Save un-resampled particles ===== #
            stacked = filter_.ssm.parameters_as_matrix(transformed=True)

            # ===== Find the best particles ===== #
            inds = self._resampler(weights, normalized=True)

            # ===== Construct distribution ===== #
            dist = self.define_pdf(stacked.concated, weights, inds)
            indep_kernel = isinstance(dist, Independent)

            # ===== Choose particles ===== #
            filter_.resample(inds)
            state.resample(inds)

            # ===== Define new filters ===== #
            prop_filt = filter_.copy((*filter_.n_parallel, 1))

            # ===== Update parameters ===== #
            rvs = dist.sample(() if indep_kernel else (stacked.concated.shape[0],))
            new_params = tuple(unflattify(rvs[:, msk], ps) for msk, ps in zip(stacked.mask, stacked.prev_shape))

            prop_filt.ssm.update_parameters(new_params)

            # ===== Calculate acceptance probabilities ===== #
            prop_state, model_loss = self.calc_model_loss(prop_filt, filter_, state)

            kernel_diff = 0.
            if not indep_kernel:
                kernel_diff += dist.log_prob(stacked.concated[inds]) - dist.log_prob(rvs)

            # ===== Check which to accept ===== #
            toaccept = torch.empty_like(model_loss).uniform_().log() < (model_loss + kernel_diff)

            # ===== Update the description ===== #
            self.accepted = toaccept.sum().float() / float(toaccept.shape[0])

            filter_.exchange(prop_filt, toaccept)
            state.exchange(prop_state, toaccept)
            weights = torch.ones_like(weights) / weights.shape[0]

        return self


class SymmetricMH(ParticleMetropolisHastings):
    def define_pdf(self, values, weights, inds):
        return _construct_mvn(values, weights, scale=1.1)   # Same scale in in particles


# Same as: https://github.com/nchopin/particles/blob/master/particles/smc_samplers.py
class AdaptiveRandomWalk(ParticleMetropolisHastings):
    def define_pdf(self, values, weights, inds):
        mvn = _construct_mvn(values, weights)
        chol = 2.38 / sqrt(values.shape[1]) * mvn.scale_tril

        return Independent(MultivariateNormal(values[inds], scale_tril=chol), 1)
