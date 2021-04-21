import torch
from abc import ABC
from ..base import BaseFilter


class BaseKalmanFilter(BaseFilter, ABC):
    def set_nparallel(self, num_filters):
        self._n_parallel = torch.tensor(num_filters)

        return self
