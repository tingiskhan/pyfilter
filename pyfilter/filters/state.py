from torch import Tensor
from torch.nn import Module


class BaseState(Module):
    def get_mean(self) -> Tensor:
        raise NotImplementedError()

    def resample(self, inds: Tensor):
        raise NotImplementedError()

    def get_loglikelihood(self) -> Tensor:
        raise NotImplementedError()

    def exchange(self, state, inds: Tensor):
        raise NotImplementedError()
