import torch
from pyro.distributions import Categorical

from .base import Proposal


# TODO: Implement for more than bootstrap...
class NestedProposal(Proposal):
    """
    Implements a nested proposal as per `Naesseth et al`_.

    .. _`Naesseth et al`: https://arxiv.org/pdf/1903.04797.pdf
    """

    def __init__(self, num_samples: int, **kwargs):
        """
        Internal initializer for :class`NestedProposal`.

        Args:
            num_samples (int): number of samples to use for nesting.
        """

        super().__init__(**kwargs)
        self._num_samples = torch.Size([num_samples])
        self._fill_val = 1.0 / num_samples

    def sample_and_weight(self, y, prediction):        
        # Construct proposal density 
        hidden_density = prediction.get_predictive_density(self._model)
        samples = hidden_density.sample(self._num_samples)
        temp_state = prediction.get_timeseries_state().propagate_from(values=samples)

        # Approximate optimal
        log_prob = self._model.build_density(temp_state).log_prob(y).nan_to_num(-torch.inf, -torch.inf)
        probs = log_prob.softmax(dim=0)

        nan_mask = probs.isnan()
        probs.masked_fill_(nan_mask, self._fill_val)
        
        best_sample = Categorical(probs.moveaxis(0, -1)).sample()

        if self._model.hidden.n_dim > 0:
            best_sample.unsqueeze_(-1)

        best_particle = samples.take_along_dim(best_sample.unsqueeze(0), dim=0).squeeze(0)
        
        return temp_state.copy(values=best_particle), log_prob.exp().mean(dim=0).log()

    def copy(self) -> "Proposal":
        return NestedProposal(self._num_samples[0])
    