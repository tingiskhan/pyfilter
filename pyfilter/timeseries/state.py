import torch
from typing import Union
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


# TODO: Might not be optimal as we construct -> deconstruct distributions and states all the time...
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

        super(JointState, self).__init__(states[0].time_index, None, None)
        self.states = torch.nn.ModuleList(states)
        self._indices = indices or self.dist.indices

    @property
    def dist(self) -> JointDistribution:
        return JointDistribution(*(s.dist for s in self.states))

    @property
    def values(self) -> torch.Tensor:
        values = tuple(
            s.values.unsqueeze(-1) if isinstance(ind, int) else s.values for s, ind in zip(self.states, self._indices)
        )

        return torch.cat(values, dim=-1)

    @values.setter
    def values(self, x):
        for s, inds in zip(self.states, self._indices):
            s.values = x[..., inds]

    def _select_dist(self, dist: Distribution, index: int) -> Distribution:
        if isinstance(dist, JointDistribution):
            return dist.distributions[index]

        if isinstance(dist, TransformedDistribution):
            assert isinstance(dist.base_dist, JointDistribution), "Cannot handle non-joint distributions!"

            transforms = list()
            # TODO: Might have to make this better...?
            for t in dist.transforms:
                inds = self._indices[index]

                to_add = t
                if isinstance(t, AffineTransform):
                    to_add = AffineTransform(t.loc[..., inds], t.scale[..., inds])

                transforms.append(to_add)

            return TransformedDistribution(dist.base_dist.distributions[index], transforms)

        raise Exception(f"Cannot handle {dist.__class__.__name__}")

    def propagate_from(self, dist: JointDistribution = None, values: torch.Tensor = None, time_increment=1.0):
        states = tuple()

        for i, (s, inds) in enumerate(zip(self.states, self._indices)):
            i_dist = self._select_dist(dist, i)
            i_values = values[..., inds] if values is not None else None

            states += (s.propagate_from(i_dist, i_values, time_increment=time_increment),)

        return JointState(*states, indices=self._indices)

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, int):
            return self.states[item]

        if not isinstance(item, slice):
            raise ValueError(f"Expected type {slice.__name__}, got {item.__class__.__name__}")

        items = range(item.start or 0, item.stop or len(self.states), item.step or 1)
        states = tuple(self.__getitem__(i) for i in items)

        if len(states) == 1:
            return states[0]

        return JointState(*states, indices=self._indices[item])
