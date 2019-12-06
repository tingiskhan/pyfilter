from ...utils import normalize
from pyfilter.algorithms.utils import stacker
import numpy as np
from ...resampling import residual


class BaseKernel(object):
    def __init__(self, record_stats=False, resampling=residual):
        """
        The base kernel used for propagating parameters.
        :param record_stats: Whether to record the statistics
        :type record_stats: bool
        """

        self._record_stats = record_stats

        self._recorded_stats = dict(
            mean=tuple(),
            scale=tuple()
        )

        self._resampler = resampling

    def set_resampler(self, resampler):
        """
        Sets the resampler to use if necessary for kernel.
        :param resampler: The resampler
        :type resampler: callable
        :rtype: BaseKernel
        """
        self._resampler = resampler

        return self

    def _update(self, parameters, filter_, weights):
        """
        Defines the function for updating the parameters for the user to override. Should return whether it resampled or
        not.
        :param parameters: The parameters of the model to update
        :type parameters: tuple[Parameter]
        :param filter_: The filter
        :type filter_: BaseFilter
        :param weights: The weights to be passed
        :type weights: torch.Tensor
        :return: Self
        :rtype: BaseKernel
        """

        raise NotImplementedError()

    def update(self, parameters, filter_, weights):
        """
        Defines the function for updating the parameters.
        :param parameters: The parameters of the model to update
        :type parameters: tuple[Parameter]
        :param filter_: The filter
        :type filter_: BaseFilter
        :param weights: The weights to be passed
        :type weights: torch.Tensor
        :return: Self
        :rtype: BaseKernel
        """

        w = normalize(weights)

        if self._record_stats:
            self.record_stats(parameters, w)

        resampled = self._update(parameters, filter_, w)
        if resampled:
            weights[:] = 0.

        return self

    def record_stats(self, parameters, weights):
        """
        Records the stats of the parameters.
        :param parameters: The parameters of the model to update
        :type parameters: tuple[Parameter]
        :param weights: The weights to be passed
        :type weights: torch.Tensor
        :return: Self
        :rtype: BaseKernel
        """

        values, _ = stacker(parameters, lambda u: u.t_values)
        weights = weights.unsqueeze(-1)

        mean = (values * weights).sum(0)
        scale = ((values - mean) ** 2 * weights).sum(0).sqrt()

        self._recorded_stats['mean'] += (mean,)
        self._recorded_stats['scale'] += (scale,)

        return self

    def get_as_numpy(self):
        """
        Returns the stats numpy arrays instead of torch tensor.
        :rtype: dict[str,np.ndarray]
        """

        res = dict()
        for k, v in self._recorded_stats.items():
            t_res = tuple()
            for pt in v:
                t_res += (pt.cpu().numpy(),)

            res[k] = np.stack(t_res)

        return res