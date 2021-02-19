import torch
from abc import ABC
from ..base import BaseFilter


class BaseKalmanFilter(BaseFilter, ABC):
    def set_nparallel(self, n):
        self._n_parallel = torch.tensor(n)

        return self
