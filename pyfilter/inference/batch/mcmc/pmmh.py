from .utils import run_pmmh, seed
from torch.distributions import Distribution
import torch
from typing import Dict, Any, Callable
from ..base import BatchFilterAlgorithm
from ....logging import LoggingWrapper
from ..state import PMMHState
from ....filters import BaseFilter, FilterResult
from .proposal import IndependentProposal


PropConstructor = Callable[[PMMHState, BaseFilter, FilterResult], Distribution]


class PMMH(BatchFilterAlgorithm):
    def __init__(self, filter_, iterations: int, num_chains: int = 4, proposal_builder: PropConstructor = None):
        """
        Implements the Particle Marginal Metropolis Hastings algorithm.
        """

        super().__init__(filter_, iterations)
        self._num_chains = num_chains
        self._proposal_builder = proposal_builder or IndependentProposal()
        self._filter_kw = {"record_states": True}

    def initialize(self, y: torch.Tensor, *args, **kwargs) -> PMMHState:
        self._filter = seed(self._filter, y, 50, self._num_chains)
        prev_res = self._filter.longfilter(y, bar=False, **self._filter_kw)

        return PMMHState(self._filter.ssm.parameters_to_array(), prev_res)

    def _fit(self, y: torch.Tensor, logging_wrapper: LoggingWrapper, **kwargs):
        state = self.initialize(y, **kwargs)

        prop_filt = self._filter.copy((*self._filter.n_parallel, 1))

        logging_wrapper.set_num_iter(self._max_iter)
        for i in range(self._max_iter):
            prop_dist = self._proposal_builder(state, self._filter, state.filter_result)
            accept, new_res, prop_filt = run_pmmh(
                self._filter,
                state.filter_result,
                prop_dist,
                prop_filt,
                y,
                **self._filter_kw
            )

            state.filter_result.exchange(new_res, accept)
            self._filter.exchange(prop_filt, accept)

            state.update(self._filter.ssm.parameters_to_array())
            logging_wrapper.do_log(i, self, y)

        return state

    def populate_state_dict(self) -> Dict[str, Any]:
        return {"_filter": self._filter.state_dict()}
