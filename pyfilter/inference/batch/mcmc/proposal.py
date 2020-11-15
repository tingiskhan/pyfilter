from torch.distributions import Independent, Normal
from ..state import PMMHState
from ....filters import BaseFilter, FilterResult


class IndependentProposal(object):
    def __init__(self, scale=1e-2):
        self._scale = scale

    def __call__(self, state: PMMHState, filter_: BaseFilter, filter_result: FilterResult):
        return Independent(Normal(filter_.ssm.parameters_to_array(True), self._scale), 1)
