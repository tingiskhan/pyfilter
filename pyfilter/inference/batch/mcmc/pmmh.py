import torch
from .utils import seed, run_pmmh
from .proposals import RandomWalk, BaseProposal
from .state import PMMHState
from ..base import BatchFilterAlgorithm
from ...utils import params_to_tensor
from ...logging import TQDMWrapper


class PMMH(BatchFilterAlgorithm):
    """
    Implements the `Particle Marginal Metropolis Hastings` algorithm found in `Particle Markov chain Monte Carlo
    methods` by C. Andrieu et al.
    """

    def __init__(self, filter_, samples: int, num_chains: int = 4, proposal: BaseProposal = None):
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
        """

        super().__init__(filter_, samples)
        self._num_chains = num_chains
        self._proposal = proposal or RandomWalk()

    def initialize(self, y: torch.Tensor, *args, **kwargs) -> PMMHState:
        self._filter = seed(self._filter, y, 50, self._num_chains)
        prev_res = self._filter.longfilter(y, bar=False)

        return PMMHState(params_to_tensor(self._filter.ssm, constrained=True), prev_res)

    def fit(self, y: torch.Tensor, logging=None, **kwargs):
        state = self.initialize(y, **kwargs)

        logging = logging or TQDMWrapper()

        try:
            logging.initialize(self, self._max_iter)
            prop_dist = self._proposal.build(state, self._filter, y)

            for i in range(self._max_iter):
                run_pmmh(self._filter, state, self._proposal, prop_dist, y, mutate_kernel=True)

                state.update_chain(params_to_tensor(self._filter.ssm, constrained=True))
                logging.do_log(i, state)

            return state

        except Exception as e:
            raise e
        finally:
            logging.teardown()
