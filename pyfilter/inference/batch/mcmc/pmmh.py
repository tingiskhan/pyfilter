from .utils import seed
import torch
from ..base import BatchFilterAlgorithm
from pyfilter.inference.logging import LoggingWrapper
from .state import PMMHState
from ...utils import PropConstructor, run_pmmh, params_to_tensor
from .proposal import IndependentProposal


class PMMH(BatchFilterAlgorithm):
    """
    Implements the Particle Marginal Metropolis Hastings algorithm.
    """

    def __init__(self, filter_, iterations: int, num_chains: int = 4, proposal_builder: PropConstructor = None):
        super().__init__(filter_, iterations)
        self._num_chains = num_chains
        self._proposal_builder = proposal_builder or IndependentProposal()
        self._filter_kw = {"record_states": True}

    def initialize(self, y: torch.Tensor, *args, **kwargs) -> PMMHState:
        self._filter = seed(self._filter, y, 50, self._num_chains)
        prev_res = self._filter.longfilter(y, bar=False, **self._filter_kw)

        return PMMHState(params_to_tensor(self._filter.ssm, constrained=True), prev_res)

    def _fit(self, y: torch.Tensor, logging_wrapper: LoggingWrapper, **kwargs):
        state = self.initialize(y, **kwargs)

        prop_filt = self._filter.copy()

        logging_wrapper.initialize(self, self._max_iter)

        for i in range(self._max_iter):
            prop_dist = self._proposal_builder(state, self._filter, y)
            accept, new_res, prop_filt = run_pmmh(
                self._filter, state.filter_result, prop_dist, prop_filt, y, **self._filter_kw
            )

            state.filter_result.exchange(new_res, accept)
            self._filter.exchange(prop_filt, accept)

            state.update(params_to_tensor(self._filter.ssm, constrained=True))
            logging_wrapper.do_log(i, state)

        return state
