import torch
from .utils import seed, run_pmmh
from .proposals import RandomWalk, BaseProposal
from .state import PMMHResult
from ..base import BatchFilterAlgorithm
from ...logging import TQDMWrapper
from ....filters import ParticleFilter


class PMMH(BatchFilterAlgorithm):
    """
    Implements the `Particle Marginal Metropolis Hastings` algorithm found in `Particle Markov chain Monte Carlo
    methods` by C. Andrieu et al.
    """

    def __init__(
        self, filter_, samples: int, num_chains: int = 4, proposal: BaseProposal = None, initializer: str = "seed"
    ):
        """
        Initializes the ``PMMH`` class.

        Args:
             filter_: See base.
             samples: The number of PMMH samples to draw.
             num_chains: The number of parallel chains to run. The total number of samples on termination is thus
                ``samples * num_chains``. Do note that we utilize broadcasting rather than ``for``-loops.
             proposal: Optional parameter specifying how to construct the proposal density for candidate
                :math:`\\theta^*` given the previously accepted candidate :math:`\\theta_i`. If not specified, defaults
                to ``pyfilter.inference.batch.mcmc.proposals.RandomWalk``.
            initializer: Optional parameter specifying how to initialize the chain:
                - ``seed``: Seeds the initial value by running several chains in parallel and choosing the one
                    maximizing the total likelihood
                - ``mean``: Sets the initial values as the means of the prior distributions. Uses MC sampling for
                    determining the mean.
        """

        super().__init__(filter_, samples)
        self._num_chains = num_chains
        self._proposal = proposal or RandomWalk()
        self._initializer = initializer

    def initialize(self, y: torch.Tensor) -> PMMHResult:
        if self._initializer == "seed":
            init_params = seed(self.filter.copy(), y, 50, self._num_chains)
        elif self._initializer == "mean":
            self.filter.ssm.sample_params(torch.Size([5_000]))
            init_params = self.filter.ssm.concat_parameters(constrained=True).mean(0)
        else:
            raise NotImplementedError(f"``{self._initializer}`` is not configured!")

        self.filter.set_num_parallel(self._num_chains)

        size = torch.Size([self._num_chains, 1] if isinstance(self.filter, ParticleFilter) else [self._num_chains])

        self.filter.ssm.sample_params(size)
        self.filter.ssm.update_parameters_from_tensor(init_params.unsqueeze(0), constrained=True)

        prev_res = self._filter.longfilter(y, bar=False)

        return PMMHResult(self._filter.ssm.concat_parameters(constrained=True, flatten=True), prev_res)

    def fit(self, y: torch.Tensor, logging=None, **kwargs):
        state = self.initialize(y)

        logging = logging or TQDMWrapper()

        try:
            logging.initialize(self, self._max_iter)
            prop_dist = self._proposal.build(state, self._filter, y)

            for i in range(self._max_iter):
                run_pmmh(self._filter, state, self._proposal, prop_dist, y, mutate_kernel=True)

                state.update_chain(self._filter.ssm.concat_parameters(constrained=True, flatten=True))
                logging.do_log(i, state)

            return state

        except Exception as e:
            raise e
        finally:
            logging.teardown()
