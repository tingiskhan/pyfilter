from ..utils import flatten
import torch
from .base import StochasticProcess, StochasticProcessBase


class StateSpaceModel(StochasticProcessBase):
    def __init__(self, hidden, observable):
        """
        Combines a hidden and observable processes to constitute a state-space model.
        :param hidden: The hidden process(es) constituting the SSM
        :type hidden: StochasticProcess
        :param observable: The observable process(es) constituting the SSM
        :type observable: StochasticProcess
        """

        self.hidden = hidden
        self.observable = observable
        self.observable._inputdim = self.hidden_ndim

    @property
    def theta_dists(self):
        return flatten(self.hidden.theta_dists, self.observable.theta_dists)

    @property
    def hidden_ndim(self):
        """
        Returns the dimension of the hidden process.
        :return: The dimension of the hidden process
        :rtype: int
        """

        return self.hidden.ndim

    @property
    def obs_ndim(self):
        """
        Returns the dimension of the observable process
        :return: The dimension of the observable process
        :rtype: int
        """

        return self.observable.ndim

    def propagate(self, x, as_dist=False):
        return self.hidden.propagate(x, as_dist=as_dist)

    def _log_prob(self, y, x):
        return self.observable.log_prob(y, x)

    def viewify_params(self, shape):
        for mod in [self.hidden, self.observable]:
            mod.viewify_params(shape)

        return self

    def h_weight(self, y, x):
        """
        Weights the process of the current hidden state `x_t`, with the previous `x_{t-1}`.
        :param y: The current hidden state
        :type y: torch.Tensor
        :param x: The previous hidden state
        :type x: torch.Tensor
        :return: The corresponding log-weights
        :rtype: torch.Tensor
        """

        return self.hidden.log_prob(y, x)

    def sample_path(self, steps, x_s=None, samples=None):
        x = x_s if x_s is not None else self.hidden.i_sample(shape=samples)

        hidden = self.hidden.sample_path(steps, x_s=x)
        obs = self.observable.propagate(hidden)

        return hidden, obs

    def exchange(self, indices, newmodel):
        """
        Exchanges the parameters of `self` with `newmodel` at indices.
        :param indices: The indices to exchange
        :type indices: torch.Tensor
        :param newmodel: The model which to exchange with
        :type newmodel: StateSpaceModel
        :return: Self
        :rtype: StateSpaceModel
        """

        procs = (
            (newmodel.hidden.theta_dists, self.hidden.theta_dists),
            (newmodel.observable.theta_dists, self.observable.theta_dists)
        )

        for proc in procs:
            for newp, oldp in zip(*proc):
                oldp.values[indices] = newp.values[indices]

        return self