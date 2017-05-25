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