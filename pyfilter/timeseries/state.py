from torch.nn import Module
import torch
from typing import Union
from torch.distributions import Distribution


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
        return NewState(self.time_index, dist, values=values)

    def propagate_from(self, dist: Distribution = None, values: torch.Tensor = None, time_increment=1.0):
        return NewState(self.time_index + time_increment, dist, values)
