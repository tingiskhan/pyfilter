import torch
from typing import Union, Optional, Sequence
from torch.distributions import Distribution
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

        self.dist = distribution

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

    def copy(self, dist: Distribution = None, values: torch.Tensor = None, time_index=None) -> "NewState":
        """
        Returns a new instance of ``NewState`` with specified ``dist`` and ``values`` but with ``time_index`` of
        current instance.

        Args:
            dist: See ``__init__``.
            values: See ``__init__``.
            time_index: See ``__init__``.
        """

        res = self.propagate_from(dist, values, time_increment=0.0)

        if time_index is not None:
           res._time_index = time_index

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


class JointState(NewState):
    """
    State object for ``JointStochasticProcess``.
    """

    # TODO: Add support for names of slices?
    def __init__(self, *args, indices: Sequence[Union[int, slice]] = None, original_states: Sequence[NewState] = None, **kwargs):
        """
        Initializes the ``JointState`` class.

        Args:
            args: See base.
            indices: See ``pyfilter.distributions.JointDistribution``.
        """

        super().__init__(*args, **kwargs)

        if indices is None and self.dist is None:
            raise ValueError("Both ``mask`` and ``dist`` cannot be None!")

        self.indices = indices or self.dist.indices
        self.original_states = tuple(s.copy() for s in original_states) or tuple()

    @classmethod
    def from_states(cls, *states: NewState, indices: Sequence[Union[int, slice]] = None) -> "JointState":
        """
        Given a sequence of ``NewState`` construct a ``JointState`` object.

        Args:
            states: An iterable of states to combine into an instance of ``JointState``.
            indices: See ``__init__``.
        """

        return JointState(
            time_index=cls._join_timeindex(*states),
            values=cls._join_values(*states),
            distribution=cls._join_distributions(*states, indices=indices),
            indices=indices,
            original_states=tuple(s.copy() for s in states)
        )

    @staticmethod
    def _join_values(*states: NewState) -> Optional[torch.Tensor]:
        if all(s._values is None for s in states):
            return None

        to_concat = tuple(s.values for s in states)
        return torch.stack(to_concat, dim=-1)

    @staticmethod
    def _join_distributions(*states: NewState, indices=None) -> Optional[JointDistribution]:
        if all(s.dist is None for s in states):
            return None

        return JointDistribution(*(s.dist for s in states), indices=indices)

    @staticmethod
    def _join_timeindex(*states: NewState) -> torch.Tensor:
        return states[0].time_index

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, int):
            dist = self.dist.distributions[item] if isinstance(self.dist, JointDistribution) else None
            values = self.values[..., self.indices[item]]

            # TODO: Do we need to enforce? Think so...
            return self.original_states[item].copy(dist=dist, values=values, time_index=self.time_index)

        if not isinstance(item, slice):
            raise ValueError(f"Expected type {slice.__name__}, got {item.__class__.__name__}")

        items = range(item.start or 0, item.stop or len(self.indices), item.step or 1)
        states = tuple(self.__getitem__(i) for i in items)

        if len(states) == 1:
            return states[0]

        return self.from_states(*states, indices=self.indices[item])

    def propagate_from(self, dist: Distribution = None, values: torch.Tensor = None, time_increment=1.0):
        return JointState(
            time_index=self.time_index + time_increment,
            distribution=dist,
            values=values,
            indices=self.indices,
            original_states=self.original_states
        )
