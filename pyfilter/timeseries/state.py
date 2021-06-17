from torch.nn import Module
import torch
from typing import Union, Optional
from torch.distributions import Distribution
from ..distributions import JointDistribution


# TODO: Rename to TimeseriesState/ProcessState
class NewState(Module):
    """
    The state object for timeseries.
    """

    def __init__(
        self, time_index: Union[float, torch.Tensor], distribution: Distribution = None, values: torch.Tensor = None
    ):
        super().__init__()

        if distribution is None and values is None:
            raise Exception("Both `distribution` and `values` cannot be `None`!")

        self.time_index = time_index if isinstance(time_index, torch.Tensor) else torch.tensor(time_index)
        self.dist = distribution
        self._values = values

    @property
    def values(self) -> torch.Tensor:
        if self._values is not None:
            return self._values

        self._values = self.dist.sample()
        return self._values

    @values.setter
    def values(self, x):
        self._values = x

    @property
    def shape(self):
        return self.values.shape

    @property
    def device(self):
        return self.values.device

    def copy(self, dist: Distribution = None, values: torch.Tensor = None):
        return NewState(self.time_index, dist, values)

    def propagate_from(self, dist: Distribution = None, values: torch.Tensor = None, time_increment=1.0):
        return NewState(self.time_index + time_increment, dist, values)


class JointState(NewState):
    """
    Implements an object for handling joint states.
    """

    def __init__(self, *args, mask=None, **kwargs):
        super().__init__(*args, **kwargs)

        if mask is None and self.dist is None:
            raise ValueError("Both `mask` and `dist` cannot be None!")

        self.mask = mask or self.dist.masks

    @classmethod
    def from_states(cls, *states, mask=None):
        return JointState(
            time_index=cls._join_timeindex(*states),
            values=cls._join_values(*states),
            distribution=cls._join_distributions(*states, mask=mask),
        )

    @staticmethod
    def _join_values(*states) -> Optional[torch.Tensor]:
        if all(s._values is None for s in states):
            return None

        to_concat = tuple(s.values.unsqueeze(-1) if len(s.dist.event_shape) == 0 else s.values for s in states)
        return torch.cat(to_concat)

    @staticmethod
    def _join_distributions(*states, mask=None) -> Optional[JointDistribution]:
        if all(s.dist is None for s in states):
            return None

        return JointDistribution(*(s.dist for s in states), masks=mask)

    # TODO: Should perhaps be first available?
    @staticmethod
    def _join_timeindex(*states) -> torch.Tensor:
        return torch.stack(tuple(s.time_index for s in states), dim=-1)

    # TODO: Joint of joint states does not work (don't really see the use case, but might be worth fixing)
    def __getitem__(self, item: int):
        return NewState(
            time_index=self.time_index[item],
            distribution=self.dist.distributions[item] if isinstance(self.dist, JointDistribution) else None,
            values=self.values[..., self.mask[item]],
        )

    def propagate_from(self, dist: Distribution = None, values: torch.Tensor = None, time_increment=1.0):
        return JointState(time_index=self.time_index + time_increment, distribution=dist, values=values, mask=self.mask)

    def copy(self, dist: Distribution = None, values: torch.Tensor = None):
        return JointState(time_index=self.time_index, distribution=dist, values=values, mask=self.mask)
