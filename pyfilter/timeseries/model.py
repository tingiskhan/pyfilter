import torch
from torch.nn import Module
from .process import StochasticProcess
from typing import Tuple


class StateSpaceModel(Module):
    """
    Combines a hidden and observable processes to constitute a state space model.
    """

    def __init__(self, hidden: StochasticProcess, observable: StochasticProcess):
        super().__init__()
        self.hidden = hidden
        self.observable = observable
        self.observable._input_dim = self.hidden_ndim

    @property
    def hidden_ndim(self) -> int:
        return self.hidden.n_dim

    @property
    def obs_ndim(self) -> int:
        return self.observable.n_dim

    def parameters_to_array(self, constrained=False, as_tuple=False):
        hidden = self.hidden.parameters_to_array(constrained, True)
        obs = self.observable.parameters_to_array(constrained, True)

        tot = hidden + obs

        if not tot or as_tuple:
            return tot

        return torch.cat(tot, dim=-1)

    def parameters_from_array(self, array, constrained=False):
        hid_shape = sum(p.get_numel(constrained) for p in self.hidden.priors())

        self.hidden.parameters_from_array(array[..., :hid_shape], constrained=constrained)
        self.observable.parameters_from_array(array[..., hid_shape:], constrained=constrained)

        return self

    def sample_path(self, steps, samples=None, x_s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_s if x_s is not None else self.hidden.initial_sample(shape=samples)

        hidden = tuple()
        obs = tuple()

        for t in range(steps):
            hidden += (x,)
            obs += (self.observable.propagate(x),)

            x = self.hidden.propagate(x)

        return torch.stack([t.state for t in hidden]), torch.stack([t.state for t in obs])

    def exchange(self, indices: torch.Tensor, new_model: "StateSpaceModel"):
        """
        Exchanges the parameters of `self` with `newmodel` at indices.

        :param indices: The indices to exchange
        :param new_model: The model which to exchange with
        """

        for self_proc, new_proc in [(self.hidden, new_model.hidden), (self.observable, new_model.observable)]:
            for new_param, self_param in zip(new_proc.parameters(), self_proc.parameters()):
                self_param[indices] = new_param[indices]

        return self
