import torch
from abc import ABC
from ..base import BaseFilter


class BaseKalmanFilter(BaseFilter, ABC):
    """
    Abstract base class for Kalman type filters.
    """

    def set_num_parallel(self, num_filters):
        self._n_parallel = torch.tensor(num_filters)

        return self
