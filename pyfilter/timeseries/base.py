from abc import ABC
from torch.distributions import Distribution
import torch
from functools import lru_cache
from .parameter import Parameter, size_getter
from ..utils import HelperMixin
from copy import deepcopy
from .utils import tensor_caster, tensor_caster_mult


class StochasticProcessBase(HelperMixin):
    @property
    def theta(self):
        """
        Returns the parameters of the model.
        :rtype: tuple[Parameter]
        """

        raise NotImplementedError()

    @property
    def theta_dists(self):
        """
        Returns the parameters that are distributions.
        :rtype: tuple[Parameter]
        """

        raise NotImplementedError()

    def viewify_params(self, shape):
        """
        Makes views of the parameters.
        :param shape: The shape to use. Please note that this shape will be prepended to the "event shape"
        :type shape: tuple|torch.Size
        :return: Self
        :rtype: StochasticProcessBase
        """

        raise NotImplementedError()

    def sample_params(self, shape=None):
        """
        Samples the parameters of the model in place.
        :param shape: The shape to use
        :return: Self
        :rtype: StochasticProcessBase
        """

        for param in self.theta_dists:
            param.sample_(shape)

        self.viewify_params(shape)

        return self

    @tensor_caster_mult
    def log_prob(self, y, x):
        """
        Weights the process of the current state `x_t` with the previous `x_{t-1}`. Used whenever the proposal
        distribution is different from the underlying.
        :param y: The value at x_t
        :type y: torch.Tensor
        :param x: The value at x_{t-1}
        :type x: torch.Tensor
        :return: The log-weights
        :rtype: torch.Tensor
        """

        return self._log_prob(y, x)

    def _log_prob(self, y, x):
        raise NotImplementedError()

    @tensor_caster
    def propagate(self, x, as_dist=False):
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous state
        :type x: torch.Tensor
        :param as_dist: Whether to return the new value as a distribution
        :type as_dist: bool
        :return: Samples from the model
        :rtype: torch.Tensor|Distribution
        """

        return self._propagate(x, as_dist)

    def _propagate(self, x, as_dist=False):
        raise NotImplementedError()

    def sample_path(self, steps, samples=None):
        """
        Samples a trajectory from the model.
        :param steps: The number of steps
        :type steps: int
        :param samples: Number of sample paths
        :type samples: int
        :return: An array of sampled values
        :rtype: torch.Tensor
        """

        raise NotImplementedError()

    def p_apply(self, func, transformed=False):
        """
        Applies `func` to each parameter of the model "inplace", i.e. manipulates `self.theta`.
        :param func: Function to apply, must be of the structure func(param)
        :type func: callable
        :param transformed: Whether or not results from applied function are transformed variables
        :type transformed: bool
        :return: Instance of self
        :rtype: StochasticProcessBase
        """

        for p in self.theta_dists:
            if transformed:
                p.t_values = func(p)
            else:
                p.values = func(p)

        return self

    def p_prior(self, transformed=True):
        """
        Calculates the prior log-likelihood of the current values.
        :param transformed: If you use an unconstrained proposal you need to use `transformed=True`
        :type transformed: bool
        :return: The prior of the current parameter values
        :rtype: torch.Tensor
        """

        if transformed:
            prop1 = 'transformed_dist'
            prop2 = 't_values'
        else:
            prop1 = 'dist'
            prop2 = 'values'

        return sum(self.p_map(lambda u: getattr(u, prop1).log_prob(getattr(u, prop2))))

    def p_map(self, func):
        """
        Applies the func to the parameters and returns a tuple of objects. Note that it is only applied to parameters
        that are distributions.
        :param func: The function to apply to parameters.
        :type func: callable
        :return: Returns tuple of values
        :rtype: tuple[Parameter]
        """

        out = tuple()
        for p in self.theta_dists:
            out += (func(p),)

        return out

    def copy(self):
        """
        Returns a deep copy of the object.
        :return: Copy of current instance
        :rtype: StochasticProcessBase
        """

        return deepcopy(self)


class StochasticProcess(StochasticProcessBase, ABC):
    def __init__(self, theta, initial_dist, increment_dist):
        """
        The base class for time series.
        :param theta: The parameters governing the dynamics
        :type theta: tuple[Distribution]|tuple[torch.Tensor]|tuple[float]
        :param initial_dist: The initial distribution
        :type initial_dist: Distribution
        :param increment_dist: The distribution of the increments
        :type increment_dist: Distribution
        """

        super().__init__()

        # ===== Check distributions ===== #
        cases = (
            all(isinstance(n, Distribution) for n in (initial_dist, increment_dist)),
            (isinstance(increment_dist, Distribution) and initial_dist is None)
        )

        if not any(cases):
            raise ValueError('All must be of instance `torch.distributions.Distribution`!')

        self.initial_dist = initial_dist
        self.increment_dist = increment_dist

        # ===== Some helpers ===== #
        self._theta = None
        self._theta_vals = None
        self._viewshape = None

        self._inputdim = self.ndim
        self._event_dim = 0 if self.ndim < 2 else 1

        # ===== Parameters ===== #
        self._dist_theta = dict()
        # TODO: Make sure same keys are same reference
        for n in [self.initial_dist, self.increment_dist]:
            if n is None:
                continue

            for k, v in n.__dict__.items():
                if k.startswith('_'):
                    continue

                if isinstance(v, Parameter) and n is self.increment_dist:
                    self._dist_theta[k] = v

        self.theta = tuple(Parameter(th) if not isinstance(th, Parameter) else th for th in theta)

        # ===== Check dimensions ===== #
        self._verify_dimensions()

    def _verify_dimensions(self):
        """
        Helper method for verifying that all return values are congruent.
        :return: Self
        :rtype: AffineModel
        """

        # TODO: Implement
        return self

    @property
    def distributional_theta(self):
        """
        Returns the parameters of the distribution to re-initialize the distribution with. Mainly a helper for when
        the user passes distributions parameterized by priors.
        :rtype: dict[str, Parameter]
        """

        return self._dist_theta

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, x):
        if self._theta is not None and len(x) != len(self._theta):
            raise ValueError('The number of parameters must be same!')
        if not all(isinstance(p, Parameter) for p in x):
            raise ValueError(f'Not all items are of instance {Parameter.__class__.__name__}')

        self._theta = x
        self.viewify_params(self._viewshape)

    @property
    def theta_dists(self):
        return tuple(p for p in self.theta if p.trainable) + tuple(self._dist_theta.values())

    @property
    def theta_vals(self):
        """
        Returns the values of the parameters.
        :rtype: tuple[torch.Tensor]
        """
        return self._theta_vals

    @property
    @lru_cache()
    def ndim(self):
        """
        Returns the dimension of the process.
        :return: Dimension of process
        :rtype: int
        """
        shape = self.increment_dist.event_shape
        if len(shape) < 1:
            return 1

        if len(shape) > 1:
            raise Exception('Timeseries model can at most be 1 dimensional (i.e. vector)!')

        return tuple(shape)[-1]

    def viewify_params(self, shape):
        shape = size_getter(shape)
        self._viewshape = shape

        # ===== Regular parameters ===== #
        params = tuple()
        for param in self.theta:
            if param.trainable:
                var = param.view(*shape, *param._prior.event_shape) if len(shape) > 0 else param.view(param.shape)
            else:
                var = param

            params += (var,)

        self._theta_vals = params

        # ===== Distributional parameters ===== #
        pdict = dict()
        for k, v in self.distributional_theta.items():
            pdict[k] = v.view(*shape, *v._prior.event_shape) if len(shape) > 0 else v.view(v.shape)

        if len(pdict) > 0:
            self.initial_dist.__init__(**pdict)
            self.increment_dist.__init__(**pdict)

        return self

    def i_sample(self, shape=None, as_dist=False):
        """
        Samples from the initial distribution.
        :param shape: The number of samples
        :type shape: int|tuple[int]
        :param as_dist: Whether to return the new value as a distribution
        :type as_dist: bool
        :return: Samples from the initial distribution
        :rtype: torch.Tensor
        """

        dist = self.initial_dist.expand(size_getter(shape))

        if as_dist:
            return dist

        return dist.sample()

    def sample_path(self, steps, samples=None, x_s=None):
        x_s = self.i_sample(samples) if x_s is None else x_s
        out = torch.zeros(steps, *x_s.shape, device=x_s.device, dtype=x_s.dtype)
        out[0] = x_s

        for i in range(1, steps):
            out[i] = self.propagate(out[i-1])

        return out

    @tensor_caster
    def propagate_u(self, x, u):
        """
        Propagate the process conditional on both state and draws from incremental distribution.
        :param x: The previous state
        :type x: torch.Tensor
        :param u: The current draws from the incremental distribution
        :type u: torch.Tensor
        :rtype: torch.Tensor
        """

        return self._propagate_u(x, u)

    def _propagate_u(self, x, u):
        raise NotImplementedError()
