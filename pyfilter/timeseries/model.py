import copy
from ..distributions.continuous import Distribution


def _get_params(parameters):
    """
    Returns the indices of the tunable parameters of `process`.
    :param parameters: The parameters
    :type parameters: tuple of Distribution
    :return: The indices
    :rtype: tuple of int
    """

    return list(i for i, p in enumerate(parameters) if isinstance(p, Distribution))


class StateSpaceModel(object):
    def __init__(self, hidden, observable):
        """
        Combines a hidden and observable processes to constitute a state-space model.
        :param hidden: The hidden process(es) constituting the SSM
        :type hidden: pyfilter.timeseries.meta.Base
        :param observable: The observable process(es) constituting the SSM
        :type observable: pyfilter.timeseries.meta.Base
        """

        self.hidden = hidden
        self.observable = observable
        self._optbounds = tuple(p.opt_bounds() for p in observable.priors), tuple(p.opt_bounds() for p in hidden.priors)

    @property
    def optbounds(self):
        """
        Returns the parameter bounds
        :rtype: tuple of tuple
        """

        return self._optbounds

    @property
    def ind_hiddenparams(self):
        """
        Returns the indices of the hidden parameters able to tune.
        :rtype: tuple of int
        """

        return _get_params(self.hidden.priors)

    @property
    def ind_obsparams(self):
        """
        Returns the indices of the hidden parameters able to tune.
        :rtype: tuple of int
        """

        return _get_params(self.observable.priors)

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

    def initialize(self, size=None, **kwargs):
        """
        Initializes the algorithm and samples from the hidden densities.
        :param size: The number of samples to for estimation of the hidden state
        :type size: int|tuple of int
        :return: An array of sampled values according to the hidden process' distribution
        :rtype: np.ndarray|float|int
        """

        return self.hidden.i_sample(size, **kwargs)

    def propagate(self, x):
        """
        Propagates the state conditional on the previous state, and parameters.
        :param x: Previous state
        :type x: np.ndarray|float|int
        :return: Next sampled state
        :rtype: np.ndarray|float|int
        """

        return self.hidden.propagate(x)

    def weight(self, y, x, params=None):
        """
        Weights the model using the current observation `y` and the current state `x`.
        :param y: The current observation
        :type y: np.ndarray|float|int
        :param x: The current state
        :type x: np.ndarray|float|int
        :param params: Whether to override the current set of parameters
        :type params: tuple of np.ndarray|tuple of float|tuple of int
        :return: The corresponding log-weights
        :rtype: np.ndarray|float|int
        """

        return self.observable.weight(y, x, params)

    def h_weight(self, y, x, params=None):
        """
        Weights the process of the current hidden state `x_t`, with the previous `x_{t-1}`.
        :param y: The current hidden state
        :type y: np.ndarray|float|int
        :param x: The previous hidden state
        :type x: np.ndarray|float|int
        :param params: Whether to override the current set of parameters
        :type params: tuple of np.ndarray|tuple of float|tuple of int
        :return: The corresponding log-weights
        :rtype: np.ndarray|float|int
        """

        return self.hidden.weight(y, x, params)

    def sample(self, steps, x_s=None):
        """
        Constructs a sample trajectory for both the observable and the hidden process.
        :param steps: The number of steps
        :type steps: int
        :param x_s: The starting value
        :type x_s: np.ndarray|float|int
        :return: Sampled trajectories
        :rtype: tuple of list
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
        :type x: np.ndarray|float|int
        :return: The mean of the next sample
        :rtype: np.ndarray|float|int
        """

        return self.hidden.mean(x)

    def copy(self):
        """
        Returns a copy of the model.
        :return: Copy of current instance
        :rtype: StateSpaceModel
        """

        return copy.deepcopy(self)

    def p_apply(self, func):
        """
        Applies `func` to each of the parameters of the model "inplace", i.e. manipulates `self.theta` of hidden and
        observable process.
        :param func: Function to apply, must be of the structure func(param).
        :type func: callable
        :return: Self
        :rtype: StateSpaceModel
        """

        self.hidden.p_apply(func)
        self.observable.p_apply(func)

        return self

    def p_prior(self):
        """
        Calculates the prior likelihood of current values of parameters.
        :return: The prior evaluated at current parameter values
        :rtype: np.ndarray|float
        """

        return self.hidden.p_prior() + self.observable.p_prior()

    def p_map(self, func, **kwargs):
        """
        Applies func to each of the parameters of the model and returns result as tuple.
        :param func: Function to apply, must of structure func(params).
        :type func: callable
        :param kwargs: Additional key-worded arguments passed to `p_map` of hidden/observable
        :rtype: tuple of tuple of np.ndarray|float
        """

        return self.hidden.p_map(func, **kwargs), self.observable.p_map(func, **kwargs)

    def p_grad(self, y, x, xo, h=1e-3):
        """
        Calculates the gradient of the parameters at the current values.
        :param y: The current observation
        :type y: np.ndarray|float|int
        :param x: The current state
        :type x: np.ndarray|float|int
        :param xo: The previous state
        :type xo: np.ndarray|float|int
        :param h: The finite difference approximation to use.
        :type h: float
        :return: Tuple of gradient estimates for each parameter of the hidden/observable
        :rtype: tuple of tuple of np.ndarray|float
        """

        return self.hidden.p_grad(x, xo, h=h), self.observable.p_grad(y, x, h=h)

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

        # ===== Exchange hidden parameters ===== #

        for i, param in enumerate(self.hidden.priors):
            if isinstance(param, Distribution):
                self.hidden.theta[i][indices] = newmodel.hidden.theta[i][indices]

        # ===== Exchange observable parameters ====== #

        for i, param in enumerate(self.observable.priors):
            if isinstance(param, Distribution):
                self.observable.theta[i][indices] = newmodel.observable.theta[i][indices]

        return self