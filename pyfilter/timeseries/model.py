import copy
from pyfilter.distributions.continuous import Distribution


class StateSpaceModel(object):
    def __init__(self, hidden, observable):
        """
        Combines a hidden and observable processes to constitute a state-space model.
        :param hidden: The hidden process(es) constituting the SSM
        :type hidden: tuple of pyfilter.timeseries.meta.Base|pyfilter.timeseries.meta.Base
        :param observable: The observable process(es) constituting the SSM
        :type observable: pyfilter.timeseries.meta.Base
        """

        self.hidden = hidden
        self.observable = observable

    @property
    def hidden_ndim(self):
        """
        Returns the dimension of the hidden process.
        :return:
        """

        return self.hidden.ndim

    @property
    def obs_ndim(self):
        """
        Returns the dimension of the observable process
        :return:
        """

        return self.observable.ndim

    def initialize(self, size=None, **kwargs):
        """
        Initializes the algorithm and samples from the hidden densities.
        :param size: The number of samples to use
        :return:
        """

        return self.hidden.i_sample(size, **kwargs)

    def propagate(self, x):
        """
        Propagates the state conditional on the previous one and the parameters.
        :param x: Previous states
        :return:
        """

        return self.hidden.propagate(x)

    def weight(self, y, x):
        """
        Weights the model using the current observation y and the current observation x.
        :param y: The current observation
        :param x: The current state
        :return:
        """

        return self.observable.weight(y, x)

    def h_weight(self, y, x):
        """
        Weights the process of the current hidden state x_t, with the previous x_{t-1}.
        :param y: The current hidden state.
        :param x: The previous hidden state.
        :return:
        """

        return self.hidden.weight(y, x)

    def sample(self, steps, x_s=None):
        """
        Constructs a sample trajectory for both the observable and the hidden density.
        :param steps: The number of steps
        :param x_s: The starting value
        :return:
        """

        hidden, obs = list(), list()

        x = x_s if x_s is not None else self.initialize()
        y = self.observable.propagate(x)

        hidden.append(x)
        obs.append(y)

        for i in range(1, steps):
            x = self.propagate(x)
            y = self.observable.propagate(x)

            hidden.append(x)
            obs.append(y)

        return hidden, obs

    def propagate_apf(self, x):
        """
        Propagates one step ahead using the mean of the hidden timeseries distribution - used in the APF.
        :param x: Previous states
        :return:
        """

        return self.hidden.mean(x)

    def copy(self):
        """
        Returns a copy of the model.
        :return: 
        """

        return copy.deepcopy(self)

    def p_apply(self, func):
        """
        Applies func to each of the parameters of the model.
        :param func: Function to apply, must be of the structure func(param).
        :return: 
        """

        self.hidden.p_apply(func)
        self.observable.p_apply(func)

        return self

    def p_prior(self):
        """
        Calculates the prior likelihood of current values of parameters.
        :return:
        """

        return self.hidden.p_prior() + self.observable.p_prior()

    def p_map(self, func, **kwargs):
        """
        Applies func to each of the parameters of the model and returns result as tuple.
        :param func: Function to apply, must of structure func(params).
        :return:
        """

        return self.hidden.p_map(func, **kwargs), self.observable.p_map(func, **kwargs)

    def p_grad(self, y, x, xo, h=1e-3):
        """
        Calculates the gradient of the parameters at the current values.
        :param y: The current observation
        :param x: The current state
        :param xo: The previous state.
        :param h: The finite difference approximation to use.
        :return:
        """

        return self.hidden.p_grad(x, xo, h=h), self.observable.p_grad(y, x, h=h)

    def exchange(self, indices, newmodel):
        """
        Exchanges the parameters of `self` with `newmodel` at indices.
        :param indices: The indices to exchange
        :type indices: np.ndarray
        :param newmodel: The model which to exchange with
        :type newmodel: StateSpaceModel
        :return:
        """

        # ===== Exchange hidden parameters ===== #

        for i, param in enumerate(self.hidden.priors):
            if isinstance(param, Distribution):
                self.hidden.theta[i][indices] = newmodel.hidden.theta[i][indices]

        # ===== Exchange observable parameters ====== #

        for i, param in enumerate(self.observable.priors):
            if isinstance(param, Distribution):
                self.observable.theta[i][indices] = newmodel.observable.theta[i][indices]

        return self