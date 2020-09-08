import torch
from ..module import TensorContainerBase, TensorContainer


def enforce_tensor(func):
    def wrapper(obj, y, *args, **kwargs):
        if not isinstance(y, torch.Tensor):
            raise ValueError('The observation must be of type Tensor!')

        return func(obj, y, *args, **kwargs)

    return wrapper


def _construct_empty(array: torch.Tensor) -> torch.Tensor:
    """
    Constructs an empty array based on the shape.
    :param array: The array to reshape after
    """

    temp = torch.arange(array.shape[-1], device=array.device)
    return temp * torch.ones_like(array, dtype=temp.dtype)


class FilterResult(TensorContainerBase):
    def __init__(self):
        """
        Implements a basic object for storing log likelihoods and the filtered means of a filter.
        """
        super().__init__()

        self._loglikelihood = None    # type: torch.Tensor
        self._filter_means = TensorContainer()

    @property
    def tensors(self):
        if not isinstance(self._loglikelihood, torch.Tensor):
            return self._filter_means.tensors

        return (self._loglikelihood,) + self._filter_means.tensors

    @property
    def loglikelihood(self) -> torch.Tensor:
        return self._loglikelihood

    @property
    def filter_means(self) -> torch.Tensor:
        if len(self._filter_means) > 0:
            return torch.stack(self._filter_means.tensors)

        return torch.empty(0)

    def exchange(self, res, inds: torch.Tensor):
        """
        Exchanges the specified indices of self with res.
        :param res: The other filter result
        :type res: FilterResult
        :param inds: The indices
        """

        # ===== Loglikelihood ===== #
        self._loglikelihood[inds] = res.loglikelihood[inds]

        # ===== Filter means ====== #
        if len(self._filter_means) > 0:
            old_fm = self.filter_means
            old_fm[:, inds] = res.filter_means[:, inds]

            self._filter_means = TensorContainer(*old_fm)

        return self

    def resample(self, inds: torch.Tensor):
        """
        Resamples the specified indices of self with res.
        :param inds: The indices
        """

        self._loglikelihood = self.loglikelihood[inds]

        if len(self._filter_means) > 0:
            self._filter_means = TensorContainer(*self.filter_means[:, inds])

        return self

    def append(self, xm, ll):
        self._filter_means.append(xm)

        if self._loglikelihood is None:
            self._loglikelihood = torch.zeros_like(ll)

        self._loglikelihood += ll

        return self