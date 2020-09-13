import torch
from .base import BatchFilterAlgorithm
from .state import PMMHState
from torch.distributions import MultivariateNormal
from ...utils import unflattify


# TODO: Add "initialize with"
class RandomWalkMetropolis(BatchFilterAlgorithm):
    def __init__(self, filter_, samples=2000, n_chains=8, initialize_with: str = None):
        """
        Implements the random walk Metropolis Hastings algorithm.
        :param samples: The number of samples to produce
        :param n_chains: The number of parallel chains to run
        :param initialize_with: Whether to use another algorithm to initialize with
        """

        super().__init__(filter_)
        self._samples = samples
        self._npar = n_chains
        self._initialize_with = initialize_with

    def _seed_initial_value(self, y: torch.Tensor) -> torch.Tensor:
        while True and self._initialize_with is None:
            self.filter.reset().ssm.sample_params(self._npar).viewify_params((self._npar, 1))
            self.filter.longfilter(y, bar=False)

            if torch.isfinite(self.filter.result.loglikelihood).all():
                return 1e-4 * torch.eye(len(self.filter.ssm.theta_dists))

    def _fit(self, y, logging_wrapper, **kwargs):
        # ===== Initialize model ===== #
        self.filter.set_nparallel(self._npar)
        self.filter.ssm.sample_params(self._npar)

        # ===== Seed stuff ===== #
        cov = self._seed_initial_value(y)

        # ===== Initialize parameters ==== #
        stacked = self.filter.ssm.parameters_as_matrix(transformed=True)
        state = PMMHState(stacked.concated, self._npar)

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
            t_filt.viewify_params((self._npar, 1))

            # ===== Calculate acceptance ===== #
            t_filt.longfilter(y, bar=False)

            logl_diff = t_filt.result.loglikelihood - self.filter.result.loglikelihood
            prior_diff = t_filt.ssm.p_prior(True) - self.filter.ssm.p_prior(True)

            accepted = logl_diff + prior_diff > torch.empty(self._npar).uniform_().log()

            # ===== Check acceptance ===== #
            self.filter.exchange(t_filt, accepted)

            stacked = self.filter.ssm.parameters_as_matrix()
            state.accepted += accepted.float().sum()

            # ===== Update state ===== #
            state.update(stacked.concated)
            logging_wrapper.do_log(i, self, y)

        return state


