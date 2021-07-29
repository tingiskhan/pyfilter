import torch
from .utils import seed, run_pmmh
from .proposals import RandomWalk, BaseProposal
from .state import PMMHState
from ..base import BatchFilterAlgorithm
from ...utils import params_to_tensor
from ...logging import TQDMWrapper


class PMMH(BatchFilterAlgorithm):
    """
    Implements the Particle Marginal Metropolis Hastings algorithm.
    """

    def __init__(self, filter_, iterations: int, num_chains: int = 4, proposal: BaseProposal = None):
        super().__init__(filter_, iterations)
        self._num_chains = num_chains
        self._proposal = proposal or RandomWalk()
        self.filter.record_states = True

    def initialize(self, y: torch.Tensor, *args, **kwargs) -> PMMHState:
        self._filter = seed(self._filter, y, 50, self._num_chains)
        prev_res = self._filter.longfilter(y, bar=False)

        return PMMHState(params_to_tensor(self._filter.ssm, constrained=True), prev_res)

    def fit(self, y: torch.Tensor, logging=None, **kwargs):
        state = self.initialize(y, **kwargs)

        prop_filt = self._filter.copy()

        logging = logging or TQDMWrapper()

        try:
            logging.initialize(self, self._max_iter)
            prop_dist = self._proposal.build(state, self._filter, y)

            for i in range(self._max_iter):
                accept, new_res, prop_filt = run_pmmh(self._filter, state, self._proposal, prop_dist, prop_filt, y)

                state.filter_state.exchange(new_res, accept)
                self._filter.exchange(prop_filt, accept)

                state.update(params_to_tensor(self._filter.ssm, constrained=True))
                logging.do_log(i, state)

            return state

        except Exception as e:
            raise e
        finally:
            logging.close()
