import torch
from typing import Union, Optional
from torch.distributions import Distribution, TransformedDistribution, AffineTransform
from ..distributions import JointDistribution
from ..state import BaseState


# TODO: Rename to TimeseriesState/ProcessState
# TODO: Would be nice to serialize distributions...
# TODO: Add step to ensure that ``values`` are serialized. Just sample on get perhaps and skip lazy eval?
class NewState(BaseState):
    """
    State object for ``StochasticProcess``.
    """

    def __init__(
        self, time_index: Union[float, torch.Tensor], distribution: Distribution = None, values: torch.Tensor = None,
    ):
        """
        Initializes the ``NewState`` class.

        Args:
            time_index: The time index of the state.
            distribution: Optional parameter, the distribution of the state at ``time_index``.
            values: Optional parameter, the values of the state at ``time_index``. If ``None`` and passing
                ``distribution`` values will be sampled from ``distribution`` when accessing ``.values`` attribute.
        """

        super().__init__()

        self._dist = distribution

        self.register_buffer(
            "_time_index", time_index if isinstance(time_index, torch.Tensor) else torch.tensor(time_index)
        )
        self.register_buffer("_values", values)
        self.register_buffer("_exog", None)

    @property
    def time_index(self) -> torch.Tensor:
        """
        The time index of the state.
        """

        return self._time_index

    @property
    def values(self) -> torch.Tensor:
        """
        The values of the state.
        """

        if self._values is not None:
            return self._values

        self._values = self.dist.sample()
        return self._values

    @values.setter
    def values(self, x):
        self._values = x

    @property
    def dist(self) -> Distribution:
        return self._dist

    @property
    def exog(self) -> torch.Tensor:
        """
        Returns the exogenous variable (if any).
        """

        return self._exog

    @exog.setter
    def exog(self, x: torch.Tensor):
        self._exog = x

    @property
    def shape(self):
        """
        The shape of ``.values``.
        """

        return self.values.shape

    @property
    def device(self):
        """
        The device of ``.values``.
        """

        return self.values.device

    def copy(self, dist: Distribution = None, values: torch.Tensor = None) -> "NewState":
        """
        Returns a new instance of ``NewState`` with specified ``dist`` and ``values`` but with ``time_index`` of
        current instance.

        Args:
            dist: See ``__init__``.
            values: See ``__init__``.
        """

        res = self.propagate_from(dist, values, time_increment=0.0)
        res.add_exog(self.exog)

        return res

    def propagate_from(self, dist: Distribution = None, values: torch.Tensor = None, time_increment=1.0):
        """
        Returns a new instance of ``NewState`` with ``dist`` and ``values``, and ``time_index`` given by
        ``.time_index + time_increment``.

        Args:
            dist: See ``__init__``.
            values: See ``__init__``.
            time_increment: Optional, specifies how much to increase ``.time_index`` with for new state.
        """

        return NewState(self.time_index + time_increment, dist, values)

    def add_exog(self, x: torch.Tensor):
        """
        Adds an exogenous variable to the state.

        Args:
            x: The exogenous variable.
        """

        self.exog = x


# TODO: FIX THIS
class JointState(NewState):
    """
    State object for ``JointStochasticProcess``.
    """

    def __init__(self, *states: Union[NewState, "JointState"], indices=None):
        """
        Initializes the ``JointState`` class.

        Args:
            states: The states to concatenate.
        """

        dist = self._join_distributions(*states, indices=indices)
        super(JointState, self).__init__(states[0].time_index, dist, self._join_values(*states))

        self.states = torch.nn.ModuleList(states)
        self._indices = indices or dist.indices

    @staticmethod
    def _join_values(*states: NewState) -> Optional[torch.Tensor]:
        if all(s._values is None for s in states):
            return None

        to_concat = tuple(s.values.unsqueeze(-1) if len(s.dist.event_shape) == 0 else s.values for s in states)
        return torch.cat(to_concat)

    @staticmethod
    def _join_distributions(*states: NewState, indices=None) -> Optional[JointDistribution]:
        if all(s.dist is None for s in states):
            return None

        return JointDistribution(*(s.dist for s in states), indices=indices)

    def _select_dist(self, dist: Distribution, index: int):
        if isinstance(dist, JointDistribution):
            return dist.distributions[index]

        if isinstance(dist, TransformedDistribution) and isinstance(dist.base_dist, JointDistribution):
            sub_transforms = list()

            for t in dist.transforms:
                to_add = t
                if isinstance(t, AffineTransform):
                    inds = self._indices[index]
                    to_add = AffineTransform(t.loc[..., inds], t.scale[..., inds])

                sub_transforms.append(to_add)

            return TransformedDistribution(dist.base_dist.distributions[index], sub_transforms)

        return None

    def _select_values(self, values: torch.Tensor, index):
        if values is None:
            return None

        return values[..., self._indices[index]]

    def propagate_from(self, dist: Distribution = None, values: torch.Tensor = None, time_increment=1.0):
        new_sub_states = map(
            lambda u: u.propagate_from(None, None, time_increment=time_increment), self.states
        )

        res = JointState(*new_sub_states, indices=self._indices)
        res._dist = dist
        res._values = values

        return res

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.states[item].copy(
                dist=self._select_dist(self.dist, item), values=self._select_values(self.values, item)
            )

        if not isinstance(item, slice):
            raise ValueError(f"Expected type {slice.__name__}, got {item.__class__.__name__}")

        items = range(item.start or 0, item.stop or len(self.states), item.step or 1)
        states = tuple(self.__getitem__(i) for i in items)

        if len(states) == 1:
            return states[0]

        return JointState(*states, indices=self._indices[item])
