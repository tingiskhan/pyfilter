import copy


class StateSpaceModel(object):
    def __init__(self, hidden, observable):
        """
        Combines a hidden and observable processes to constitute a state-space model.
        :param hidden: The hidden process(es) constituting the SSM
        :type hidden: tuple of pyfilter.timeseries.meta.Base|pyfilter.timeseries.meta.Base
        :param observable: The observable process(es) constituting the SSM
        :type observable: pyfilter.timeseries.meta.Base
        """

        self.hidden = hidden if isinstance(hidden, tuple) else (hidden,)
        self.observable = observable

    @property
    def hidden_ndim(self):
        """
        Returns the dimension of the hidden process.
        :return:
        """

        return tuple(h.ndim for h in self.hidden)

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

        out = list()
        for ts in self.hidden:
            out.append(ts.i_sample(size, **kwargs))

        return out

    def propagate(self, x):
        """
        Propagates the state conditional on the previous one and the parameters.
        :param x: Previous states
        :return:
        """

        out = list()
        for ts, xi in zip(self.hidden, x):
            out.append(ts.propagate(xi))

        return out

    def weight(self, y, x):
        """
        Weights the model using the current observation y and the current observation x.
        :param y: The current observation
        :param x: The current state
        :return:
        """

        return self.observable.weight(y, *x)

    def h_weight(self, y, x):
        """
        Weights the process of the current hidden state x_t, with the previous x_{t-1}.
        :param y: The current hidden state.
        :param x: The previous hidden state.
        :return:
        """

        return sum(h.weight(y[i], x[i]) for i, h in enumerate(self.hidden))

    def sample(self, steps, x_s=None, **kwargs):
        """
        Constructs a sample trajectory for both the observable and the hidden density.
        :param steps: The number of steps
        :param x_s: The starting value
        :param kwargs: Any kwargs
        :return:
        """

        hidden, obs = list(), list()

        x = x_s if x_s is not None else self.initialize()
        y = self.observable.propagate(*x, **kwargs)

        hidden.append(x)
        obs.append(y)

        for i in range(1, steps):
            x = self.propagate(x)
            y = self.observable.propagate(*x)

            hidden.append(x)
            obs.append(y)

        return hidden, obs

    def propagate_apf(self, x):
        """
        Propagates one step ahead using the mean of the hidden timeseries distribution - used in the APF.
        :param x: Previous states
        :return:
        """

        out = list()
        for ts, xi in zip(self.hidden, x):
            out.append(ts.mean(xi))

        return out

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

        for ts in self.hidden:
            ts.p_apply(func)

        self.observable.p_apply(func)

        return self

    def p_prior(self):
        """
        Calculates the prior likelihood of current values of parameters.
        :return:
        """

        out = 0
        for ts in self.hidden:
            out += ts.p_prior()

        return out + self.observable.p_prior()

    def p_map(self, func, **kwargs):
        """
        Applies func to each of the parameters of the model and returns result as tuple.
        :param func: Function to apply, must of structure func(params).
        :return:
        """

        h_out = tuple()
        for ts in self.hidden:
            h_out += ts.p_map(func, **kwargs)

        return h_out, self.observable.p_map(func, **kwargs)