import torch
from .base import BatchFilterAlgorithm
from .state import PMMHState
from torch.distributions import MultivariateNormal
from ...utils import unflattify, StackedObject
from ..sequential import SMC2
from ...logging import DefaultLogger
from ..utils import _construct_mvn
from ...normalization import normalize
from typing import Tuple


# TODO: Add "initialize with"
class RandomWalkMetropolis(BatchFilterAlgorithm):
    def __init__(self, filter_, samples=500, n_chains=4, initialize_with: str = "smc2"):
        """
        Implements the random walk Metropolis Hastings algorithm.
        :param samples: The number of samples to produce
        :param n_chains: The number of parallel chains to run
        :param initialize_with: Whether to use another algorithm to initialize with
        """

        super().__init__(filter_)
        self._samples = samples
        self._chains = n_chains
        self._initialize_with = initialize_with

    def _seed_initial_value(self, y: torch.Tensor) -> Tuple[StackedObject, torch.Tensor]:
        filter_copy = self.filter.copy()

        while True and self._initialize_with is None:
            filter_copy.reset().viewify_params((self._chains, 1))
            filter_copy.longfilter(y, bar=False)

            if torch.isfinite(filter_copy.result.loglikelihood).all():
                return filter_copy.ssm.parameters_as_matrix(), 1e-4 * torch.eye(len(self.filter.ssm.theta_dists))

            filter_copy.ssm.sample_params(self._chains)

        if self._initialize_with.lower() == "smc2":
            alg = SMC2(filter_copy, 400, threshold=0.5)
            state = alg.fit(y[:300], logging=DefaultLogger())

            stacked = alg.filter.ssm.parameters_as_matrix()

            # TODO: Resample instead
            mvn = _construct_mvn(stacked.concated, normalize(state.w))
            stacked.concated = stacked.concated[:self._chains]

            return stacked, mvn.covariance_matrix

    def _fit(self, y, logging_wrapper, **kwargs):
        # ===== Initialize model ===== #
        self.filter.set_nparallel(self._chains)
        self.filter.ssm.sample_params(self._chains)

        # ===== Seed stuff ===== #
        stacked, cov = self._seed_initial_value(y)

        # ===== Initialize parameters ==== #
        new_params = tuple(unflattify(stacked.concated[..., msk], ps) for msk, ps in zip(stacked.mask, stacked.prev_shape))
        self.filter.ssm.update_parameters(new_params)
        self.filter.viewify_params((self._chains, 1))

        # ===== Run filter ===== #
        self.filter.longfilter(y, bar=False)

        # ===== Initialize state ===== #
        state = PMMHState(stacked.concated, self._chains)
        logging_wrapper.set_num_iter(self._samples)

        for i in range(1, self._samples):
            # ===== Copy filter ====== #
            t_filt = self.filter.copy().reset()

            # ===== Sample parameters ===== #
            mvn = MultivariateNormal(state.samples[-1], cov)
            rvs = mvn.sample()

            # ===== Update parameters ===== #
            new_params = tuple(unflattify(rvs[..., msk], ps) for msk, ps in zip(stacked.mask, stacked.prev_shape))
            t_filt.ssm.update_parameters(new_params)
            t_filt.viewify_params((self._chains, 1))

            # ===== Calculate acceptance ===== #
            t_filt.longfilter(y, bar=False)

            logl_diff = t_filt.result.loglikelihood - self.filter.result.loglikelihood
            prior_diff = t_filt.ssm.p_prior(True) - self.filter.ssm.p_prior(True)

            accepted = logl_diff + prior_diff > torch.empty(self._chains).uniform_().log()

            # ===== Check acceptance ===== #
            self.filter.exchange(t_filt, accepted)

            stacked = self.filter.ssm.parameters_as_matrix()
            state.accepted += accepted.float().sum()

            # ===== Update state ===== #
            state.update(stacked.concated)
            logging_wrapper.do_log(i, self, y)

        return state


