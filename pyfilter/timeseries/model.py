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
    def trainable_parameters(self):
        return self.hidden.trainable_parameters + self.observable.trainable_parameters

    @property
    def hidden_ndim(self) -> int:
        return self.hidden.ndim

    @property
    def obs_ndim(self) -> int:
        return self.observable.ndim

    def propagate(self, x, u=None, as_dist=False):
        return self.hidden.propagate(x, u=u, as_dist=as_dist)

    def log_prob(self, y, x, u=None):
        return self.observable.log_prob(y, x, u=u)

    def viewify_params(self, shape, in_place=True) -> Tuple[Tuple[Parameter, ...], ...]:
        return tuple(ssm.viewify_params(shape, in_place=in_place) for ssm in [self.hidden, self.observable])

    def parameters_to_array(self, transformed=False, as_tuple=False):
        hidden = self.hidden.parameters_to_array(transformed, True)
        obs = self.observable.parameters_to_array(transformed, True)

        tot = hidden + obs

        if not tot or as_tuple:
            return tot

        return torch.cat(tot, dim=-1)

    def parameters_from_array(self, array, transformed=False):
        hid_shape = sum(p.numel_(transformed) for p in self.hidden.trainable_parameters)

        self.hidden.parameters_from_array(array[..., :hid_shape], transformed=transformed)
        self.observable.parameters_from_array(array[..., hid_shape:], transformed=transformed)

        return self

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
            (newmodel.hidden.trainable_parameters, self.hidden.trainable_parameters),
            (newmodel.observable.trainable_parameters, self.observable.trainable_parameters),
        )

        for proc in procs:
            for new_param, self_param in zip(*proc):
                self_param.values[indices] = new_param.values[indices]

        return self

    def populate_state_dict(self):
        return {"hidden": self.hidden.state_dict(), "observable": self.observable.state_dict()}
