import copy
from abc import ABC
from ..proposals import LinearGaussianObservations
from ..resampling import systematic
from ..proposals.bootstrap import Bootstrap, Proposal
from ..timeseries import StateSpaceModel, LinearGaussianObservations as LGO
from tqdm import tqdm
import torch
from ..utils import get_ess, choose, normalize
from ..module import Module, TensorContainer
from .utils import enforce_tensor, FilterResult, _construct_empty
from typing import Tuple, Union


class BaseFilter(Module, ABC):
    def __init__(self, model: StateSpaceModel):
        """
        The basis for filters. Take as input a model and specific attributes.
        :param model: The model
        """

        self._dummy = torch.tensor(0.)

        super().__init__()

        if not isinstance(model, StateSpaceModel):
            raise ValueError('`model` must be `{:s}`!'.format(StateSpaceModel.__name__))

        self._model = model
        self._n_parallel = None
        self._filter_res = FilterResult()

    @property
    def result(self) -> FilterResult:
        """
        Returns the filtering result object.
        """

        return self._filter_res

    @property
    def ssm(self) -> StateSpaceModel:
        """
        Returns the SSM as an object.
        """
        return self._model

    def viewify_params(self, shape):
        """
        Defines views to be used as parameters instead
        :param shape: The shape to use. Please note that
        :type shape: tuple|torch.Size
        :return: Self
        """

        self.ssm.viewify_params(shape)

        return self

    def set_nparallel(self, n: int):
        """
        Sets the number of parallel filters to use
        :param n: The number of parallel filters
        """

        raise NotImplementedError()

    def initialize(self):
        """
        Initializes the filter.
        :return: Self
        """

        return self

    @enforce_tensor
    def filter(self, y: Union[float, torch.Tensor]):
        """
        Performs a filtering the model for the observation `y`.
        :param y: The observation
        :return: Self
        """

        self._filter_res.append(*self._filter(y))

        return self

    def _filter(self, y: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The actual filtering procedure. Overwrite this.
        :param y: The observation
        :return: Mean of state, log-likelihood
        """

        raise NotImplementedError()

    def longfilter(self, y: Union[torch.Tensor, Tuple[torch.Tensor, ...]], bar=True):
        """
        Filters the entire data set `y`.
        :param y: An array of data. Should be {# observations, # dimensions (minimum of 1)}
        :param bar: Whether to print a progressbar
        :return: Self
        """

        astuple = tuple(y) if not isinstance(y, tuple) else y
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
        """

        # TODO: Need to fix the reference to _theta_vals here. If there are parameters we need to redefine them
        return copy.deepcopy(self)

    def predict(self, steps: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts `steps` ahead using the latest available information.
        :param steps: The number of steps forward to predict
        :param kwargs: Any key worded arguments
        """

        raise NotImplementedError()

    def resample(self, inds: torch.Tensor, entire_history: bool = False):
        """
        Resamples the filter, used in cases where we use nested filters.
        :param inds: The indices
        :param entire_history: Whether to resample entire history
        :return: Self
        """
        if entire_history:
            self._filter_res.resample(inds)

        # ===== Resample the parameters of the model ====== #
        self.ssm.p_apply(lambda u: choose(u.values, inds))
        self._resample(inds)

        return self

    def _resample(self, inds: torch.Tensor):
        """
        Implements resampling unique for the filter.
        :param inds: The indices
        :return: Self
        """

        return self

    def reset(self):
        """
        Resets the filter by nullifying the filter specific attributes.
        :return: Self
        """

        self._filter_res = FilterResult()
        self._reset()

        return self

    def _reset(self):
        """
        Any filter specific resets.
        :return: Self
        """

        return self

    def exchange(self, filter_, inds: torch.Tensor):
        """
        Exchanges the filters.
        :param filter_: The new filter
        :type filter_: BaseFilter
        :param inds: The indices
        :return: Self
        """

        self._model.exchange(inds, filter_.ssm)
        self._filter_res.exchange(filter_._filter_res, inds)
        self._exchange(filter_, inds)

        return self

    def _exchange(self, filter_, inds: torch.Tensor):
        """
        Filter specific exchanges.
        :param filter_: The new filter
        :type filter_: BaseFilter
        :param inds: The indices
        :return: Self
        """

        return self


_PROPOSAL_MAPPING = {
    LGO: LinearGaussianObservations
}


class ParticleFilter(BaseFilter, ABC):
    def __init__(self, model, particles: int, resampling=systematic, proposal: Union[str, Proposal] = 'auto', ess=0.9,
                 need_grad=False):
        """
        Implements the base functionality of a particle filter.
        :param particles: How many particles to use
        :param resampling: Which resampling method to use
        :param proposal: Which proposal to use, set to `auto` to let algorithm decide
        :param ess: At which level to resample
        :param need_grad: Whether we need the gradient
        """

        super().__init__(model)

        self.particles = particles
        self._th = ess

        # ===== State variables ===== #
        self._x_cur = None                          # type: torch.Tensor
        self._inds = None                           # type: torch.Tensor
        self._w_old = None                          # type: torch.Tensor

        # ===== Auxiliary variable ===== #
        self._sumaxis = -(1 + self.ssm.hidden_ndim)
        self._rsample = need_grad

        # ===== Resampling function ===== #
        self._resampler = resampling

        # ===== Logged ESS ===== #
        self.logged_ess = TensorContainer()

        # ===== Proposal ===== #
        if proposal == 'auto':
            try:
                proposal = _PROPOSAL_MAPPING[type(self._model)]()
            except KeyError:
                proposal = Bootstrap()

        self._proposal = proposal.set_model(self._model)    # type: Proposal

    @property
    def particles(self) -> torch.Size:
        """
        Returns the number of particles.
        """

        return self._particles

    @particles.setter
    def particles(self, x: Tuple[int, int] or int):
        """
        Sets the number of particles.
        """

        self._particles = torch.Size([x]) if not isinstance(x, (tuple, list)) else torch.Size(x)

    @property
    def proposal(self) -> Proposal:
        """
        Returns the proposal.
        """

        return self._proposal

    def _resample_state(self, w: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, bool]]:
        """
        Resamples the state in accordance with the weigths.
        :param w: The weights
        :return: The indices and mask
        """

        # ===== Get the ones requiring resampling ====== #
        ess = get_ess(w) / w.shape[-1]
        mask = ess < self._th

        self.logged_ess.append(ess)

        # ===== Create a default array for resampling ===== #
        out = _construct_empty(w)

        # ===== Return based on if it's nested or not ===== #
        if not mask.any():
            return out, mask
        elif not isinstance(self._particles, tuple):
            return self._resampler(w), mask

        out[mask] = self._resampler(w[mask])

        return out, mask

    def set_nparallel(self, n: int):
        if len(self.particles) > 1:
            raise NotImplementedError('Currently only supports at most one level of nesting!')

        self._n_parallel = torch.Size([n])
        self.particles = (*self._n_parallel, *self.particles)

        if self._x_cur is not None:
            return self.initialize()

        return self

    def initialize(self):
        self._x_cur = self._model.hidden.i_sample(self.particles)
        self._w_old = torch.zeros(self.particles, device=self._x_cur.device)

        return self

    def predict(self, steps, aggregate: bool = True, **kwargs):
        x, y = self._model.sample_path(steps + 1, x_s=self._x_cur, **kwargs)

        if not aggregate:
            return x[1:], y[1:]

        w = normalize(self._w_old)
        wsqd = w.unsqueeze(-1)

        xm = (x * (wsqd if self.ssm.hidden_ndim > 1 else w)).sum(self._sumaxis)
        ym = (y * (wsqd if self.ssm.obs_ndim > 1 else w)).sum(-2 if self.ssm.obs_ndim > 1 else -1)

        return xm[1:], ym[1:]

    def _resample(self, inds):
        self._x_cur = self._x_cur[inds]
        self._w_old = self._w_old[inds]

        return self

    def _exchange(self, filter_, inds):
        self._x_cur[inds] = filter_._x_cur[inds]
        self._w_old[inds] = filter_._w_old[inds]

        return self

    def _reset(self):
        self.logged_ess = TensorContainer()
        return self


class BaseKalmanFilter(BaseFilter, ABC):
    def set_nparallel(self, n):
        self._n_parallel = torch.Size([n])

        return self