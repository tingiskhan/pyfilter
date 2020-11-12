from pyfilter.utils import normalize
import numpy as np
from ....resampling import systematic
import torch
from typing import Iterable, Callable, Union, Dict
from ....timeseries import Parameter
from ....filters import BaseFilter, FilterResult


def finite_decorator(func):
    def wrapper(obj, parameters, filter_, state, w):
        mask = ~torch.isfinite(w)
        w[mask] = -float("inf")

        return func(obj, parameters, filter_, state, w)

    return wrapper


class BaseKernel(object):
    def __init__(self, record_stats=False, resampling=systematic):
        """
        The base kernel used for updating the collection of particles approximating the posterior.
        """

        self._record_stats = record_stats

        self._recorded_stats = dict(mean=tuple(), scale=tuple())

        self._resampler = resampling

    def set_resampler(self, resampler: Callable[[torch.Tensor, bool, Union[float, torch.Tensor]], torch.Tensor]):
        self._resampler = resampler

        return self

    def _update(self, parameters: Iterable[Parameter], filter_: BaseFilter, state: FilterResult, weights: torch.Tensor):
        raise NotImplementedError()

    @finite_decorator
    def update(self, parameters: Iterable[Parameter], filter_: BaseFilter, state: FilterResult, weights: torch.Tensor):
        """
        Defines the function for updating the parameters.
        :param parameters: The parameters of the model to update
        :param filter_: The filter
        :param state: The previous state of the filter
        :param weights: The weights to use for propagating.
        :return: Self
        """

        w = normalize(weights)

        if self._record_stats:
            self.record_stats(parameters, w)

        self._update(parameters, filter_, state, w)

        return self

    def record_stats(self, parameters: Iterable[Parameter], weights: torch.Tensor):
        stacked = torch.cat(tuple(p.t_values.view(-1, p.numel_(True)) for p in parameters), -1)
        weights = weights.unsqueeze(-1)

        mean = (stacked * weights).sum(0)
        scale = ((stacked - mean) ** 2 * weights).sum(0).sqrt()

        self._recorded_stats["mean"] += (mean,)
        self._recorded_stats["scale"] += (scale,)

        return self

    def get_as_numpy(self) -> Dict[str, np.ndarray]:
        res = dict()
        for k, v in self._recorded_stats.items():
            t_res = tuple()
            for pt in v:
                t_res += (pt.cpu().numpy(),)

            res[k] = np.stack(t_res)

        return res
