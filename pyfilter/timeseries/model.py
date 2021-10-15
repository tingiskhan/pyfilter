import torch
from torch.nn import Module
from typing import Tuple
from copy import deepcopy
from .stochasticprocess import StochasticProcess
from ..prior_module import UpdateParametersMixin, HasPriorsModule


# TODO: Add methods ``concat_parameters`` and ``update_parameters_from_tensor``
class StateSpaceModel(Module, UpdateParametersMixin):
    """
    Class representing a state space model, i.e. a dynamical system given by the pair stochastic processes
    :math:`\\{X_t\\}` and :math:`\\{Y_t\\}`, where :math:`X_t` is independent from :math:`Y_t`, and :math:`Y_t`
    conditionally indpendent given :math:`X_t`. See more `here`_.

    .. _`here`: https://en.wikipedia.org/wiki/State-space_representation
    """

    def __init__(self, hidden: StochasticProcess, observable: StochasticProcess):
        """
        Initializes the ``StateSpaceModel`` class.

        Args:
            hidden: The hidden process.
            observable: The observable process.
        """

        super().__init__()
        self.hidden = hidden
        self.observable = observable

    def sample_path(self, steps, samples=None, x_s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_s if x_s is not None else self.hidden.initial_sample(shape=samples)

        hidden = (x,)
        obs = tuple()

        for t in range(1, steps + 1):
            x = self.hidden.propagate(x)

            obs_state = self.observable.propagate(x)

            obs += (obs_state,)
            hidden += (x,)

        return torch.stack([t.values for t in hidden]), torch.stack([t.values for t in obs])

    def exchange(self, indices: torch.Tensor, other: "StateSpaceModel") -> "StateSpaceModel":
        """
        Exchanges the parameters of ``self`` with ``other`` at ``indices``.

        Args:
            indices: The indices at which exchange parameters.
            other: The other model to exchange parameters with.
        """

        for self_proc, new_proc in [(self.hidden, other.hidden), (self.observable, other.observable)]:
            for new_param, self_param in zip(new_proc.parameters(), self_proc.parameters()):
                self_param[indices] = new_param[indices]

        return self

    def copy(self) -> "StateSpaceModel":
        """
        Creates a deep copy of ``self``.
        """

        return deepcopy(self)

    def sample_params(self, shape: torch.Size = torch.Size([])):
        # TODO: Convert this check to either a wrapper or on instantiation
        for m in (self.hidden, self.observable):
            if not issubclass(m.__class__, HasPriorsModule):
                raise Exception(f"``{m}`` must be subclassed by {HasPriorsModule.__name__}!")

            m.sample_params(shape)

    def concat_parameters(self, constrained=False, flatten=True) -> torch.Tensor:
        res = tuple()
        for m in (self.hidden, self.observable):
            if not issubclass(m.__class__, HasPriorsModule):
                raise Exception(f"``{m}`` must be subclassed by {HasPriorsModule.__name__}!")

            res += (m.concat_parameters(constrained, flatten),)

        return torch.cat(res, dim=-1)

    def update_parameters_from_tensor(self, x: torch.Tensor, constrained=False):

        hidden_priors_elem = sum(prior.get_numel(constrained) for prior in self.hidden.priors())

        hidden_x = x[..., :hidden_priors_elem]
        observable_x = x[..., hidden_priors_elem:]

        self.hidden.update_parameters_from_tensor(hidden_x, constrained)
        self.observable.update_parameters_from_tensor(observable_x, constrained)
