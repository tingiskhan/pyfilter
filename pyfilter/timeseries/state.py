from torch.nn import Module
import torch
from typing import Union
from torch.distributions import Distribution


# TODO: Deprecate when done with NewState
class TimeseriesState(Module):
    def __init__(self, time_index: Union[float, torch.Tensor], state: torch.Tensor):
        super(TimeseriesState, self).__init__()
        self.register_buffer(
            "_time_index", time_index if isinstance(time_index, torch.Tensor) else torch.tensor(time_index)
        )
        self.register_buffer("_state", state)

    @property
    def time_index(self) -> torch.Tensor:
        return self._buffers["_time_index"]

    @time_index.setter
    def time_index(self, x):
        self._buffers["_time_index"] = x

    @property
    def state(self) -> torch.Tensor:
        return self._buffers["_state"]

    @state.setter
    def state(self, x):
        self._buffers["_state"] = x

    @property
    def shape(self):
        return self.state.shape

    @property
    def device(self):
        return self.state.device

    def copy(self, new_values: torch.Tensor):
        return TimeseriesState(self.time_index, new_values)


class BatchedState(TimeseriesState):
    pass


# TODO: Rename to TimeseriesState/ProcessState
class NewState(object):
    """
    The state object for timeseries.
    """

    def __init__(self, time_index: Union[float, torch.Tensor], distribution: Distribution, values: torch.Tensor = None):
        super().__init__()
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

    def copy(self, dist: Distribution, values: torch.Tensor = None):
        return NewState(self.time_index, dist, values=values)

    def state_dict(self):
        return {"time_index": self.time_index, "_values": self._values}

    def to(self, device: str):
        self.time_index = self.time_index.to(device)

        if self._values is None:
            return

        self._values = self._values.to(device)
