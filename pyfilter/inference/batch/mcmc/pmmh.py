import torch
from .utils import run_pmmh
from .proposals import RandomWalk, BaseProposal
from .state import PMMHResult
from ...base import BaseAlgorithm
from ...logging import TQDMWrapper


class PMMH(BaseAlgorithm):
    # TODO: Add reference
    """
    Implements the `Particle Marginal Metropolis Hastings` algorithm found in `Particle Markov chain Monte Carlo
    methods` by C. Andrieu et al.
    """

    MONTE_CARLO_SAMPLES = torch.Size([10_000])

    def __init__(
        self, filter_, num_samples: int, num_chains: int = 4, proposal: BaseProposal = None, initializer: str = "seed"
    ):
        """
        Initializes the :class:`PMMH` class.

        Args:
             filter_: See base.
             num_samples: The number of PMMH samples to draw.
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

        super().__init__(filter_)
        self.num_samples = num_samples

        self._num_chains = torch.Size([num_chains])
        self._proposal = proposal or RandomWalk()
        self._initializer = initializer

    def initialize(self, y: torch.Tensor) -> PMMHResult:
        self.filter.set_batch_shape(self._num_chains)
        size = torch.Size([*self._num_chains, 1])

        if self._initializer == "seed":
            raise NotImplementedError()
        elif self._initializer == "mean":
            for p in self.filter.ssm.parameters():
                dist = p.prior.build_distribution()
                p.data = dist.sample(self.MONTE_CARLO_SAMPLES).mean(dim=0).expand(size)
        else:
            raise NotImplementedError(f"``{self._initializer}`` is not configured!")

        prev_res = self._filter.batch_filter(y, bar=False)

        return PMMHResult(self.get_parameters(), prev_res)

    def fit(self, y: torch.Tensor, logging=None, **kwargs):
        state = self.initialize(y)

        logging = logging or TQDMWrapper()

        try:
            logging.initialize(self, self.num_samples)
            prop_dist = self._proposal.build(state, self._filter, y)

            for i in range(self.num_samples):
                run_pmmh(self._filter, state, self._proposal, prop_dist, y, mutate_kernel=True)

                state.update_chain(self._filter.ssm.concat_parameters(constrained=True, flatten=True))
                logging.do_log(i, state)

            return state

        except Exception as e:
            raise e
        finally:
            logging.teardown()
