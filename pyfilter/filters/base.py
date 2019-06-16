import pandas as pd
import copy
from ..proposals import LinearGaussianObservations
from ..resampling import systematic, multinomial
from ..proposals.bootstrap import Bootstrap, Proposal
from ..timeseries import StateSpaceModel, LinearGaussianObservations as LGO
from tqdm import tqdm
import torch
from ..utils import get_ess, choose, MoveToHelper


def enforce_tensor(func):
    def wrapper(obj, y, **kwargs):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=obj._device)
        elif y.device != obj._device:
            y = y.to(obj._device)

        return func(obj, y, **kwargs)

    return wrapper


class BaseFilter(MoveToHelper):
    def __init__(self, model):
        """
        The basis for filters. Take as input a model and specific attributes.
        :param model: The model
        :type model: StateSpaceModel
        """

        super().__init__()

        if not isinstance(model, StateSpaceModel):
            raise ValueError('`model` must be `{:s}`!'.format(StateSpaceModel.__name__))

        self._model = model
        self._n_parallel = None

        # ===== Some helpers ===== #
        self.s_ll = tuple()
        self.s_mx = tuple()

    @property
    def s_loglikelihood(self):
        """
        Returns the saved loglikelihood.
        :rtype: torch.Tensor
        """

        if len(self.s_ll) > 0:
            return torch.stack(self.s_ll, dim=0)

        return torch.empty((1,))

    @s_loglikelihood.setter
    def s_loglikelihood(self, x):
        """
        Sets the loglikelihood.
        :param x: The new loglikelihood
        :type x: torch.Tensor
        """

        if not isinstance(x, torch.Tensor) or x.shape != self.s_loglikelihood.shape:
            raise ValueError('Either wrong type or wrong dimensions!')

        self.s_ll = tuple(x)

    @property
    def loglikelihood(self):
        """
        Returns the total loglikelihood
        :rtype: torch.Tensor
        """

        return sum(self.s_ll)

    @property
    def ssm(self):
        """
        Returns the SSM as an object.
        :rtype: StateSpaceModel
        """
        return self._model

    def set_nparallel(self, n):
        """
        Sets the number of parallel filters to use
        :param n: The number of parallel filters
        :type n: int
        :rtype: BaseFilter
        """

        raise ValueError()

    def initialize(self):
        """
        Initializes the filter.
        :return: Self
        :rtype: BaseFilter
        """

        return self

    @enforce_tensor
    def filter(self, y):
        """
        Performs a filtering the model for the observation `y`.
        :param y: The observation
        :type y: float|torch.Tensor
        :return: Self
        :rtype: BaseFilter
        """

        xm, ll = self._filter(y)

        self.s_mx += (xm,)
        self.s_ll += (ll,)

        return self

    def _filter(self, y):
        """
        The actual filtering procedure. Overwrite this.
        :param y: The observation
        :type y: float|torch.Tensor
        :return: Mean of state, likelihood
        :rtype: torch.Tensor, torch.Tensor
        """

        raise NotImplementedError()

    def longfilter(self, y, bar=True):
        """
        Filters the entire data set `y`.
        :param y: An array of data. Should be {# observations, # dimensions (minimum of 1)}
        :type y: pd.DataFrame|torch.Tensor
        :param bar: Whether to print a progressbar
        :type bar: bool
        :return: Self
        :rtype: BaseFilter
        """

        astuple = tuple(y)
        if bar:
            iterator = tqdm(astuple, desc=str(self.__class__.__name__))
        else:
            iterator = astuple

        for yt in iterator:
            self.filter(yt)

        return self

    def copy(self):
        """
        Returns a copy of itself.
        :return: Copy of self
        :rtype: BaseFilter
        """

        return copy.deepcopy(self)

    @property
    def filtermeans(self):
        """
        Calculates the filter means and returns a timeseries.
        :return:
        """

        if len(self.s_mx) > 0:
            return torch.stack(self.s_mx, dim=0)

        return torch.empty((1,))

    @filtermeans.setter
    def filtermeans(self, x):
        """
        Sets the filter means.
        :param x: The new filter means
        :type x: torch.Tensor
        """

        if not isinstance(x, torch.Tensor) or x.shape != self.filtermeans.shape:
            raise ValueError('Either wrong type or wrong dimensions!')

        self.s_mx = tuple(x)

    def predict(self, steps):
        """
        Predicts `steps` ahead using the latest available information.
        :param steps: The number of steps forward to predict
        :type steps: int
        :rtype: tuple[torch.Tensor]
        """
        raise NotImplementedError()

    def resample(self, inds, entire_history=False):
        """
        Resamples the filter, used in cases where we use nested filters.
        :param inds: The indices
        :type inds: torch.Tensor
        :param entire_history: Whether to resample entire history
        :type entire_history: bool
        :return: Self
        :rtype: BaseFilter
        """
        if entire_history:
            for obj, name in [(self.filtermeans, 'filtermeans'), (self.s_loglikelihood, 's_loglikelihood')]:
                setattr(self, name, obj[:, inds])

        # ===== Resample the parameters of the model ====== #
        self.ssm.p_apply(lambda u: choose(u.values, inds))
        self._resample(inds)

        return self

    def _resample(self, inds):
        """
        Implements resampling unique for the filter.
        :param inds: The indices
        :type inds: torch.Tensor
        :return: Self
        :rtype: BaseFilter
        """

        return self

    def reset(self):
        """
        Resets the filter by nullifying the filter specific attributes.
        :return: Self
        :rtype: BaseFilter
        """

        self.s_mx = tuple()
        self.s_ll = tuple()

        self._reset()

        return self

    def _reset(self):
        """
        Any filter specific resets.
        :return: Self
        :rtype: BaseFilter
        """

        return self

    def exchange(self, filter_, inds):
        """
        Exchanges the filters.
        :param filter_: The new filter
        :type filter_: BaseFilter
        :param inds: The indices
        :type inds: torch.Tensor
        :return: Self
        :rtype: BaseFilter
        """

        self._model.exchange(inds, filter_.ssm)

        for obj, name in [(self.filtermeans, 'filtermeans'), (self.s_loglikelihood, 's_loglikelihood')]:
            obj[:, inds] = getattr(filter_, name)[:, inds]

            setattr(self, name, obj)

        self._exchange(filter_, inds)

        return self

    def _exchange(self, filter_, inds):
        """
        Filter specific exchanges.
        :param filter_: The new filter
        :type filter_: BaseFilter
        :param inds: The indices
        :type inds: torch.Tensor
        :return: Self
        :rtype: BaseFilter
        """

        return self


_PROPOSAL_MAPPING = {
    LGO: LinearGaussianObservations
}


def _construct_empty(array):
    """
    Constructs an empty array based on the shape.
    :param array: The array to reshape after
    :type array: torch.Tensor
    :rtype: torch.Tensor
    """

    temp = torch.arange(array.shape[-1], device=array.device)
    return temp * torch.ones_like(array, dtype=temp.dtype)


def cudawarning(resampling):
    """
    Raises an error if you're using CUDA and have `systematic` enabled.
    :return: Nothing
    :rtype: None
    """

    if 'cuda' == torch.tensor(1.).device.type and resampling is systematic:
        msg = '`systematic` relies on `numpy`, you must use `multinomial` instead as CUDA is enabled.'
        raise ValueError(msg)


class ParticleFilter(BaseFilter):
    def __init__(self, model, particles, resampling=systematic, proposal='auto', ess=0.9):
        """
        Implements the base functionality of a particle filter.
        :param particles: How many particles to use
        :type particles: int
        :param resampling: Which resampling method to use
        :type resampling: callable
        :param proposal: Which proposal to use, set to `auto` to let algorithm decide
        :type proposal: Proposal|str
        :param ess: At which level to resample
        :type ess: float
        """

        cudawarning(resampling)

        super().__init__(model)

        self._x_cur = None                          # type: torch.Tensor
        self._inds = None                           # type: torch.Tensor
        self._particles = particles
        self._w_old = None                          # type: torch.Tensor
        self._ess = ess
        self._sumaxis = -1 if self.ssm.hidden_ndim < 2 else -2

        self._resampler = resampling

        if proposal == 'auto':
            try:
                proposal = _PROPOSAL_MAPPING[type(self._model)]()
            except KeyError:
                proposal = Bootstrap()

        self._proposal = proposal.set_model(self._model)    # type: Proposal

    @property
    def proposal(self):
        """
        Returns the proposal.
        :rtype: Proposal
        """

        return self._proposal

    @proposal.setter
    def proposal(self, x):
        """
        Sets the proposal
        :param x: The new proposal
        :type x: Proposal
        """

        if not isinstance(x, Proposal):
            raise ValueError('`x` must be {:s}!'.format(Proposal.__name__))

        self._proposal = x

    @property
    def particles(self):
        """
        Returns the particles
        :rtype: int|tuple[int]
        """

        return self._particles

    @particles.setter
    def particles(self, x):
        """
        Sets the particles
        :param x: The new number of particles
        :type x: int
        """

        self._particles = x

    def _resample_state(self, weights):
        """
        Resamples the state in accordance with the weigths.
        :param weights: The weights
        :type weights: torch.Tensor
        :return: The indices and mask
        :rtype: tuple[torch.Tensor]
        """

        # ===== Get the ones requiring resampling ====== #
        ess = get_ess(weights) / weights.shape[-1]
        mask = ess < self._ess

        # ===== Create a default array for resampling ===== #
        out = _construct_empty(weights)

        # ===== Return based on if it's nested or not ===== #
        if not mask.any():
            return out, mask
        elif not isinstance(self._particles, tuple):
            return self._resampler(weights), mask

        out[mask] = self._resampler(weights[mask])

        return out, mask

    def set_nparallel(self, n):
        self._n_parallel = n

        temp = self._particles[-1] if isinstance(self._particles, (tuple, list)) else self._particles
        if n is not None:
            self._particles = self._n_parallel, temp
        else:
            self._particles = temp

        if self._x_cur is not None:
            return self.initialize()

        return self

    def initialize(self):
        self._x_cur = self._model.initialize(self._particles)
        self._w_old = torch.zeros(self._particles)

        return self

    def predict(self, steps):
        x, y = self._model.sample(steps+1, x_s=self._x_cur)

        return x[1:], y[1:]

    def _resample(self, inds):
        self._x_cur = self._x_cur[inds]
        self._w_old = self._w_old[inds]

        return self

    def _exchange(self, filter_, inds):
        self._x_cur[inds] = filter_._x_cur[inds]
        self._w_old[inds] = filter_._w_old[inds]

        return self


class KalmanFilter(BaseFilter):
    def set_nparallel(self, n):
        self._n_parallel = n

        return self