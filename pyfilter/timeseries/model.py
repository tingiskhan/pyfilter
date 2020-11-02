from ..utils import flatten, stacker
import torch
from .base import Base
from .process import StochasticProcess
from typing import Tuple
from .parameter import Parameter


class StateSpaceModel(Base):
    def __init__(self, hidden: StochasticProcess, observable: StochasticProcess):
        """
        Combines a hidden and observable processes to constitute a state-space model.
        :param hidden: The hidden process(es) constituting the SSM
        :param observable: The observable process(es) constituting the SSM
        """

        self.hidden = hidden
        self.observable = observable
        self.observable._input_dim = self.hidden_ndim

    @property
    def parameter_distributions(self):
        return flatten(self.hidden.parameter_distributions, self.observable.parameter_distributions)

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

    def viewify_params(self, shape, in_place=True) -> Tuple[Tuple[Parameter, ...], ...]:
        return tuple(ssm.viewify_params(shape, in_place=in_place) for ssm in [self.hidden, self.observable])

    def update_parameters(self, params, transformed=True):
        num_params = len(self.hidden.parameter_distributions)

        for m, ps in [(self.hidden, params[:num_params]), (self.observable, params[num_params:])]:
            m.update_parameters(ps)

        return self

    def parameters_as_matrix(self, transformed=True):
        return stacker(self.parameter_distributions, lambda u: u.t_values if transformed else u.values)

    def h_weight(self, y: torch.Tensor, x: torch.Tensor):
        """
        Weights the process of the current hidden state `x_t`, with the previous `x_{t-1}`.
        :param y: The current hidden state
        :param x: The previous hidden state
        :return: The corresponding log-weights
        """

        return self.hidden.log_prob(y, x)

    def sample_path(self, steps, samples=None, x_s=None, u=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_s if x_s is not None else self.hidden.i_sample(shape=samples)

        hidden = self.hidden.sample_path(steps, x_s=x)
        obs = self.observable.propagate(hidden, u=u)

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
            (newmodel.hidden.parameter_distributions, self.hidden.parameter_distributions),
            (newmodel.observable.parameter_distributions, self.observable.parameter_distributions)
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