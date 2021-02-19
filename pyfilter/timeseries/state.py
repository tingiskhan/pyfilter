from torch.nn import Module
import torch
from typing import Union


class TimeseriesState(Module):
    def __init__(self, time_index: Union[float, torch.Tensor], state: torch.Tensor):
        super(TimeseriesState, self).__init__()
        self.register_buffer("_time_index", torch.tensor(time_index) if isinstance(time_index, float) else time_index)
        self.register_buffer("_state", state)

    @property
    def time_index(self) -> torch.Tensor:
        return self._buffers["_time_index"]

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
