import copy
import numpy as np
from ..utils import flatten
import torch
from .parameter import Parameter


class StateSpaceModel(object):
    def __init__(self, hidden, observable):
        """
        Combines a hidden and observable processes to constitute a state-space model.
        :param hidden: The hidden process(es) constituting the SSM
        :type hidden: pyfilter.timeseries.base.AffineModel
        :param observable: The observable process(es) constituting the SSM
        :type observable: pyfilter.timeseries.base.AffineModel
        """

        self.hidden = hidden
        self.observable = observable
        self.observable._inputdim = self.hidden_ndim

    @property
    def theta_dists(self):
        """
        Returns the tuple of parameters for both hidden and observable.
        :return: (hidden parameters, observable parameters)
        :rtype: tuple[tuple[Parameter]]
        """

        return self.hidden.theta_dists, self.observable.theta_dists

    @property
    def flat_theta_dists(self):
        """
        Returns the flattened tuple of parameters for both hidden and observable.
        :return: Parameters
        :rtype: tuple[Parameter]
        """

        return flatten(self.theta_dists)

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

    def initialize(self, size=None):
        """
        Initializes the algorithm and samples from the hidden densities.
        :param size: The number of samples to for estimation of the hidden state
        :type size: int|tuple of int
        :return: An array of sampled values according to the hidden process' distribution
        :rtype: torch.Tensor
        """

        return self.hidden.i_sample(size)

    def propagate(self, x):
        """
        Propagates the state conditional on the previous state, and parameters.
        :param x: Previous state
        :type x: torch.Tensor
        :return: Next sampled state
        :rtype: torch.Tensor
        """

        return self.hidden.propagate(x)

    def weight(self, y, x):
        """
        Weights the model using the current observation `y` and the current state `x`.
        :param y: The current observation
        :type y: torch.Tensor|float
        :param x: The current state
        :type x: torch.Tensor
        :return: The corresponding log-weights
        :rtype: torch.Tensor
        """

        return self.observable.weight(y, x)

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

        return self.hidden.weight(y, x)

    def sample(self, steps, x_s=None, samples=None):
        """
        Constructs a sample trajectory for both the observable and the hidden process.
        :param steps: The number of steps
        :type steps: int
        :param x_s: The starting value
        :type x_s: torch.Tensor|float
        :param samples: How many samples
        :type samples: tuple[int]|int
        :return: Sampled trajectories
        :rtype: tuple[torch.Tensor]
        """

        x_shape = steps, self.hidden.ndim
        y_shape = steps, self.observable.ndim

        if samples is not None:
            tmp = (*((samples,) if not isinstance(samples, (list, tuple)) else samples),)
            x_shape += tmp
            y_shape += tmp

        hidden = torch.zeros(x_shape)
        obs = torch.zeros(y_shape)

        x = x_s if x_s is not None else self.initialize(size=samples)
        y = self.observable.propagate(x)

        hidden[0] = x
        obs[0] = y

        for i in range(1, steps):
            x = self.propagate(x)
            y = self.observable.propagate(x)

            hidden[i] = x
            obs[i] = y

        return hidden, obs

    def copy(self):
        """
        Returns a copy of the model.
        :return: Copy of current instance
        :rtype: StateSpaceModel
        """

        return copy.deepcopy(self)

    def p_apply(self, func, transformed=False):
        """
        Applies `func` to each of the parameters of the model "inplace", i.e. manipulates `self.theta` of hidden and
        observable process.
        :param func: Function to apply, must be of the structure func(param), and must return a `numpy.ndarray`
        :type func: callable
        :param transformed: Whether or not results from applied function are transformed variables
        :type transformed: bool
        :return: Self
        :rtype: StateSpaceModel
        """

        self.hidden.p_apply(func, transformed=transformed)
        self.observable.p_apply(func, transformed=transformed)

        return self

    def p_prior(self, transformed=True):
        """
        Calculates the prior likelihood of current values of parameters.
        :param transformed: If you use an unconstrained proposal you need to use `transformed=True`
        :type transformed: bool
        :return: The prior evaluated at current parameter values
        :rtype: torch.Tensor|float
        """

        return self.hidden.p_prior(transformed) + self.observable.p_prior(transformed)

    def p_map(self, func):
        """
        Applies func to each of the parameters of the model and returns result as tuple.
        :param func: Function to apply, must of structure func(params).
        :type func: callable
        :rtype: tuple[tuple[Parameter]]
        """

        return self.hidden.p_map(func), self.observable.p_map(func)

    def exchange(self, indices, newmodel):
        """
        Exchanges the parameters of `self` with `newmodel` at indices.
        :param indices: The indices to exchange
        :type indices: np.ndarray
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