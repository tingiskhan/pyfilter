from .mh import ParticleMetropolisHastings
from ...filters import BaseState
from torch.distributions import Distribution
from ..utils import _construct_mvn


# TODO: Add support for replacing only part of filtered means
class BlockMetropolisHastings(ParticleMetropolisHastings):
    def __init__(self, initial_state: BaseState, prev_dist: Distribution, **kwargs):
        """
        Implements a kernel using blocked samples
        """

        super().__init__(**kwargs)
        self._init_state = initial_state
        self._prev_dist = prev_dist
        self._entire_hist = False

    def define_pdf(self, values, weights, inds):
        return _construct_mvn(values, weights)

    def calc_model_loss(self, new_filter, old_filter):
        new_state = new_filter.longfilter(self._y, bar=False, init_state=self._init_state)
        diff_logl = new_filter.result.loglikelihood - old_filter.result.loglikelihood

        new_params = new_filter.ssm.parameters_as_matrix()
        old_params = old_filter.ssm.parameters_as_matrix()

        prior_diff = self._prev_dist.log_prob(new_params.concated) - self._prev_dist.log_prob(old_params.concated)

        return new_state, diff_logl + prior_diff
