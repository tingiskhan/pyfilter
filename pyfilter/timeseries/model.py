from ..utils import flatten
import torch
from .base import StochasticProcess, StochasticProcessBase
from typing import Tuple
from .parameter import Parameter


class StateSpaceModel(StochasticProcessBase):
    def __init__(self, hidden: StochasticProcess, observable: StochasticProcess):
        """
        Combines a hidden and observable processes to constitute a state-space model.
        :param hidden: The hidden process(es) constituting the SSM
        :param observable: The observable process(es) constituting the SSM
        """

        self.hidden = hidden
        self.observable = observable
        self.observable._inputdim = self.hidden_ndim

    @property
    def theta_dists(self):
        return flatten(self.hidden.theta_dists, self.observable.theta_dists)

    @property
    def hidden_ndim(self) -> int:
        """
        Returns the dimension of the hidden process.
        """

        return self.hidden.ndim

    @property
    def obs_ndim(self) -> int:
        """
        Returns the dimension of the observable process
        """

        return self.observable.ndim

    def propagate(self, x, as_dist=False):
        return self.hidden.propagate(x, as_dist=as_dist)

    def log_prob(self, y, x):
        return self.observable.log_prob(y, x)

    def viewify_params(self, shape) -> Tuple[Tuple[Parameter, ...], ...]:
        return tuple(ssm.viewify_params(shape) for ssm in [self.hidden, self.observable])

    def h_weight(self, y: torch.Tensor, x: torch.Tensor):
        """
        Weights the process of the current hidden state `x_t`, with the previous `x_{t-1}`.
        :param y: The current hidden state
        :param x: The previous hidden state
        :return: The corresponding log-weights
        """

        return self.hidden.log_prob(y, x)

    def sample_path(self, steps, samples=None, x_s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_s if x_s is not None else self.hidden.i_sample(shape=samples)

        hidden = self.hidden.sample_path(steps, x_s=x)
        obs = self.observable.propagate(hidden)

        return hidden, obs

    def exchange(self, indices: torch.Tensor, newmodel):
        """
        Exchanges the parameters of `self` with `newmodel` at indices.
        :param indices: The indices to exchange
        :param newmodel: The model which to exchange with
        :type newmodel: StateSpaceModel
        :return: Self
        """

        procs = (
            (newmodel.hidden.theta_dists, self.hidden.theta_dists),
            (newmodel.observable.theta_dists, self.observable.theta_dists)
        )

        for proc in procs:
            for newp, oldp in zip(*proc):
                oldp.values[indices] = newp.values[indices]

        return self

    def populate_state_dict(self):
        return {
            "hidden": self.hidden.state_dict(),
            "observable": self.observable.state_dict()
        }