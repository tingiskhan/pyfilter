from .base import BaseKernel
from ...utils import _construct_mvn
import torch
from torch.distributions import Distribution, MultivariateNormal, Independent
from typing import Iterable
from math import sqrt
from ...batch.pmmh import run_pmmh


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
        parameters should be propagated in transformed space.
        """

        raise NotImplementedError()

    def _update(self, parameters, filter_, state, weights):
        prop_filt = filter_.copy((*filter_.n_parallel, 1))

        for _ in range(self._nsteps):
            # ===== Save un-resampled particles ===== #
            stacked = filter_.ssm.parameters_to_array(transformed=True)

            # ===== Find the best particles ===== #
            inds = self._resampler(weights, normalized=True)

            # ===== Construct distribution ===== #
            dist = self.define_pdf(stacked, weights, inds)

            # ===== Choose particles ===== #
            filter_.resample(inds)
            state.resample(inds)

            # ===== Update parameters ===== #
            to_accept, prop_state, prop_filt = run_pmmh(filter_, state, dist, prop_filt, self._y)

            # ===== Update the description ===== #
            self.accepted = to_accept.sum().float() / float(to_accept.shape[0])

            filter_.exchange(prop_filt, to_accept)
            state.exchange(prop_state, to_accept)
            weights = torch.ones_like(weights) / weights.shape[0]

        return self


class SymmetricMH(ParticleMetropolisHastings):
    def define_pdf(self, values, weights, inds):
        return _construct_mvn(values, weights, scale=1.1)  # Same scale in in particles


# Same as: https://github.com/nchopin/particles/blob/master/particles/smc_samplers.py
class AdaptiveRandomWalk(ParticleMetropolisHastings):
    def define_pdf(self, values, weights, inds):
        mvn = _construct_mvn(values, weights)
        chol = 2.38 / sqrt(values.shape[1]) * mvn.scale_tril

        return Independent(MultivariateNormal(values[inds], scale_tril=chol), 1)
