from torch.distributions import Distribution, AffineTransform, TransformedDistribution
import torch
from functools import lru_cache
from .parameter import Parameter, size_getter
from ..utils import concater, HelperMixin
from .statevariable import StateVariable


# TODO: Should check whether the output is a tensor, else cast as one. Needs to be done here as well as
def tensor_caster(func):
    """
    Function for helping out when it comes to multivariate models. Returns a torch.Tensor
    :param func: The function to pass
    :type func: callable
    :rtype: torch.Tensor
    """

    def wrapper(obj, x):
        if obj._inputdim > 1:
            tx = StateVariable(x)
        else:
            tx = x

        res = concater(func(obj, tx))
        if not isinstance(res, torch.Tensor):
            res = torch.ones_like(x) * res

        res.__sv = tx

        return res

    return wrapper


def init_caster(func):
    def wrapper(obj):
        res = concater(func(obj))
        if not isinstance(res, torch.Tensor):
            raise ValueError('The result must be of type `torch.Tensor`!')

        return res

    return wrapper


def finite_decorator(func):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)

        mask = torch.isfinite(out)

        if (~mask).all():
            raise ValueError('All weights seem to be `nan`, adjust your model')

        out[~mask] = float('-inf')

        return out

    return wrapper


def _get_shape(x, ndim):
    """
    Gets the shape to generate samples for.
    :param x: The tensor
    :type x: torch.Tensor|float
    :param ndim: The dimensions
    :type ndim: int
    :rtype: tuple[int]
    """

    if not isinstance(x, torch.Tensor):
        return ()

    return x.shape if ndim < 2 else x.shape[:-1]


class TimeseriesBase(HelperMixin):
    def __init__(self, theta, noise):
        """
        The base class for time series.
        :param theta: The parameters governing the dynamics
        :type theta: tuple[Distribution]|tuple[torch.Tensor]|tuple[float]
        :param noise: The noise governing the noise process
        :type noise: tuple[Distribution]
        """

        super().__init__()

        # ===== Check distributions ===== #
        cases = (
            all(isinstance(n, Distribution) for n in noise),
            (isinstance(noise[-1], Distribution) and noise[0] is None)
        )

        if not any(cases):
            raise ValueError('All must be of instance `torch.distributions.Distribution`!')

        self.noise0, self.noise = noise

        # ===== Some helpers ===== #
        self._theta_vals = None
        self._viewshape = None

        self._inputdim = self.ndim
        self._event_dim = 0 if self.ndim < 2 else 1

        # ===== Parameters ===== #
        self._dist_theta = dict()
        for n in [self.noise0, self.noise]:
            if n is None:
                continue

            for k, v in n.__dict__.items():
                if k.startswith('_'):
                    continue

                if isinstance(v, Parameter) and n is self.noise:
                    self._dist_theta[k] = v
                elif isinstance(v, Parameter) and v.trainable and n is self.noise0:
                    raise ValueError('You cannot have distributional parameters in the initial distribution!')

        self._theta = tuple(Parameter(th) for th in theta)

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
        """
        Returns the parameters of the model.
        :rtype: tuple[Parameter]
        """

        return self._theta

    @property
    def theta_dists(self):
        """
        Returns the indices for parameter are distributions.
        :rtype: tuple[Parameter]
        """

        distparams = tuple(p for p in self._dist_theta.values() if isinstance(p, Parameter))
        return tuple(p for p in self.theta if p.trainable) + distparams

    @property
    @lru_cache()
    def ndim(self):
        """
        Returns the dimension of the process.
        :return: Dimension of process
        :rtype: int
        """
        shape = self.noise.event_shape
        if len(shape) < 1:
            return 1

        if len(shape) > 1:
            raise Exception('Timeseries model can at most be 1 dimensional (i.e. vector)!')

        return tuple(shape)[-1]

    def viewify_params(self, shape):
        """
        Defines views to be used as parameters instead
        :param shape: The shape to use. Please note that this shape will be prepended to the "event shape"
        :type shape: tuple|torch.Size
        :return: Self
        :rtype: AffineModel
        """
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
            pdict[k] = v.view(*shape, *v._prior.event_shape) if len(shape) > 0 else param.view(param.shape)

        if len(pdict) > 0:
            self.noise.__init__(**pdict)

        return self

    def sample_params(self, shape=None):
        """
        Samples the parameters of the model in place.
        :param shape: The shape to use
        :return: Self
        :rtype: AffineModel
        """

        for param in self.theta_dists:
            param.sample_(shape)

        self.viewify_params(shape)

        return self

    def weight(self, y, x):
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

        raise NotImplementedError()

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

        raise NotImplementedError()

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

        raise NotImplementedError()

    def sample(self, steps, samples=None):
        """
        Samples a trajectory from the model.
        :param steps: The number of steps
        :type steps: int
        :param samples: Number of sample paths
        :type samples: int
        :return: An array of sampled values
        :rtype: torch.Tensor
        """

        x_s = self.i_sample(samples)
        out = torch.zeros(steps, *x_s.shape)
        out[0] = x_s

        for i in range(1, steps):
            out[i] = self.propagate(out[i-1])

        return out

    def p_apply(self, func, transformed=False):
        """
        Applies `func` to each parameter of the model "inplace", i.e. manipulates `self.theta`.
        :param func: Function to apply, must be of the structure func(param)
        :type func: callable
        :param transformed: Whether or not results from applied function are transformed variables
        :type transformed: bool
        :return: Instance of self
        :rtype: AffineModel
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

    # TODO: Might not be optimal, but think it will work
    def __setattr__(self, key, value):
        self.__dict__[key] = value

        if key == '_theta':
            self.viewify_params(self._viewshape)


class AffineModel(TimeseriesBase):
    def __init__(self, initial, funcs, theta, noise):
        """
        Class for defining model with affine dynamics.
        :param initial: The functions governing the initial dynamics of the process
        :type initial: tuple[callable]
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple[callable]
        :param theta: The parameters governing the dynamics
        :type theta: tuple[Distribution]|tuple[torch.Tensor]|tuple[float]
        :param noise: The noise governing the noise process
        :type noise: tuple[Distribution]
        """

        super().__init__(theta, noise)

        # ===== Dynamics ===== #
        self.f0, self.g0 = initial
        self.f, self.g = funcs

    @init_caster
    def i_mean(self):
        """
        Calculates the mean of the initial distribution.
        :return: The mean of the initial distribution
        :rtype: torch.Tensor
        """

        return self.f0(*self._theta_vals)

    @init_caster
    def i_scale(self):
        """
        Calculates the scale of the initial distribution.
        :return: The scale of the initial distribution
        :rtype: torch.Tensor
        """

        return self.g0(*self._theta_vals)

    @tensor_caster
    def f_val(self, x):
        """
        Evaluates the drift part of the process.
        :param x: The state of the process.
        :type x: torch.Tensor
        :return: The mean
        :rtype: torch.Tensor
        """

        return self.f(x, *self._theta_vals)

    @tensor_caster
    def g_val(self, x):
        """
        Evaluates the diffusion part of the process.
        :param x: The state of the process.
        :type x: torch.Tensor
        :return: The mean
        :rtype: torch.Tensor
        """

        return self.g(x, *self._theta_vals)

    def mean(self, x):
        """
        Calculates the mean of the process conditional on the previous state and current parameters.
        :param x: The state of the process.
        :type x: torch.Tensor
        :return: The mean
        :rtype: torch.Tensor
        """

        return self.f_val(x)

    def scale(self, x):
        """
        Calculates the scale of the process conditional on the current state and parameters.
        :param x: The state of the process
        :type x: torch.Tensor
        :return: The scale
        :rtype: torch.Tensor
        """

        return self.g_val(x)

    def weight(self, y, x):
        loc, scale = self.mean(x), self.scale(x)

        return self.predefined_weight(y, loc, scale)

    @finite_decorator
    def predefined_weight(self, y, loc, scale):
        """
        Helper method for weighting with loc and scale.
        :param y: The value at x_t
        :type y: torch.Tensor
        :param loc: The mean
        :type loc: torch.Tensor
        :param scale: The scale
        :type scale: torch.Tensor
        :return: The log-weights
        :rtype: torch.Tensor
        """

        dist = self._define_transdist(loc, scale)

        return dist.log_prob(y)

    def _define_transdist(self, loc, scale):
        """
        Helper method for defining the transition density
        :param loc: The mean
        :type loc: torch.Tensor
        :param scale: The scale
        :type scale: torch.Tensor
        :return: Distribution
        :rtype: Distribution
        """

        loc, scale = torch.broadcast_tensors(loc, scale)

        shape = _get_shape(loc, self.ndim)

        return TransformedDistribution(self.noise.expand(shape), AffineTransform(loc, scale, event_dim=self._event_dim))

    def i_sample(self, shape=None, as_dist=False):
        shape = size_getter(shape)

        dist = TransformedDistribution(
            self.noise0.expand(shape), AffineTransform(self.i_mean(), self.i_scale(), event_dim=self._event_dim)
        )

        if as_dist:
            return dist

        return dist.sample()

    def propagate(self, x, as_dist=False):
        dist = self._define_transdist(self.mean(x), self.scale(x))

        if as_dist:
            return dist

        return dist.sample()


class AffineObservations(AffineModel):
    def __init__(self, funcs, theta, noise):
        """
        Class for defining model with affine dynamics in the observable process.
        :param funcs: The functions governing the dynamics of the process
        :type funcs: tuple of callable
        :param theta: The parameters governing the dynamics
        :type theta: tuple[Distribution]|tuple[torch.Tensor]|tuple[float]
        :param noise: The noise governing the noise process
        :type noise: Distribution
        """
        super().__init__((None, None), funcs, theta, (None, noise))

    def sample(self, steps, samples=None):
        raise NotImplementedError("Cannot sample from Observable only!")

    def _verify_dimensions(self):
        # TODO: Implement this
        return self