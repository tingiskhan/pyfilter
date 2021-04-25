import torch
from torch.nn import Module
from typing import Tuple
from copy import deepcopy
from .stochasticprocess import StochasticProcess
from .state import NewState


class StateSpaceModel(Module):
    """
    Combines a hidden and observable processes to constitute a state space model.
    """

    def __init__(self, hidden: StochasticProcess, observable: StochasticProcess, observe_every_nth_step=1):
        super().__init__()
        self.hidden = hidden
        self.observable = observable
        self._observe_every_nth_step = observe_every_nth_step

    def generate_observation(self, state: NewState) -> bool:
        return state.time_index % self._observe_every_nth_step == 0

    def sample_path(self, steps, samples=None, x_s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_s if x_s is not None else self.hidden.initial_sample(shape=samples)

        hidden = tuple()
        obs = tuple()

        nan = float("nan")
        for t in range(steps):
            hidden += (x,)

            if self.generate_observation(x):
                obs_state = self.observable.propagate(x)
            else:
                obs_state = x.propagate_from(values=obs[-1].values * nan, time_increment=0.0)

            obs += (obs_state,)

            x = self.hidden.propagate(x)

        return torch.stack([t.values for t in hidden]), torch.stack([t.values for t in obs])

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

    def copy(self):
        return deepcopy(self)
