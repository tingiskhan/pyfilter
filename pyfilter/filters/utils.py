import torch
from ..module import TensorContainerBase, TensorContainer


def enforce_tensor(func):
    def wrapper(obj, y, **kwargs):
        if not isinstance(y, torch.Tensor):
            raise ValueError('The observation must be of type Tensor!')

        return func(obj, y, **kwargs)

    return wrapper


def _construct_empty(array):
    """
    Constructs an empty array based on the shape.
    :param array: The array to reshape after
    :type array: torch.Tensor
    :rtype: torch.Tensor
    """

    temp = torch.arange(array.shape[-1], device=array.device)
    return temp * torch.ones_like(array, dtype=temp.dtype)


class FilterResult(TensorContainerBase):
    def __init__(self):
        """
        Implements a basic object for storing likelihoods and filtered means of a filter.
        """
        super().__init__()

        self._loglikelihood = TensorContainer()
        self._filter_means = TensorContainer()

    @property
    def tensors(self):
        return self._loglikelihood.tensors + self._filter_means.tensors

    @property
    def loglikelihood(self):
        return torch.stack(self._loglikelihood.tensors)

    @property
    def filter_means(self):
        return torch.stack(self._filter_means.tensors)

    def exchange(self, res, inds):
        """
        Exchanges the specified indices of self with res.
        :param res: The other filter result
        :type res: FilterResult
        :param inds: The indices
        :type inds: torch.Tensor
        :rtype: Self
        """

        # ===== Loglikelihood ===== #
        old_ll = self.loglikelihood
        old_ll[:, inds] = res.loglikelihood[:, inds]

        self._loglikelihood = TensorContainer(*old_ll)

        # ===== Filter means ====== #
        old_fm = self.filter_means
        old_fm[:, inds] = res.filter_means[:, inds]

        self._filter_means = TensorContainer(*old_fm)

        return self

    def resample(self, inds):
        """
        Resamples the specified indices of self with res.
        :param inds: The indices
        :type inds: torch.Tensor
        :rtype: Self
        """

        self._loglikelihood = TensorContainer(*self.loglikelihood[:, inds])
        self._filter_means = TensorContainer(*self.filter_means[:, inds])

        return self

    def append(self, xm, ll):
        self._filter_means.append(xm)
        self._loglikelihood.append(ll)

        return self