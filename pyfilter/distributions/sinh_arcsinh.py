import torch
from torch.distributions import constraints, utils, Transform


class SinhArcsinhTransform(Transform):
    """
    Implements the `Sinh-arcsinh` transformation of an arbitrary random variable. Basically ports the Tensorflow
    `implementation`_.

    .. _implementation: https://github.com/tensorflow/probability/blob/v0.14.1/tensorflow_probability/python/bijectors/sinh_arcsinh.py#L36-L179
    """

    domain = constraints.real
    codomain = constraints.real

    @property
    def sign(self):
        pass

    def _call(self, x):
        return torch.sinh((torch.asinh(x) + self.skew) * self.tailweight) * self._output_scaler()

    def _inverse(self, y):
        return torch.sinh(torch.asinh(y / self._output_scaler()) / self.tailweight - self.skew)

    def log_abs_det_jacobian(self, x, y):
        first = torch.cosh((torch.arcsinh(x) + self.skew) * self.tailweight).log()
        second = self.tailweight.log()
        third = -0.5 * torch.log1p(x ** 2)

        return first + second + third + self._output_scaler().log()

    def __init__(self, skew, tailweight):
        """
        Initializes the ``NormalSinhArcsinh`` transform.

        Args:
            skew: Controls the skew of the distribution.
            tailweight: Controls the kurtosis of the distribution.
        """

        super().__init__()
        self.skew, self.tailweight = utils.broadcast_all(skew, tailweight)
        self._scale = torch.tensor(2.0, dtype=self.skew.dtype)

    def _output_scaler(self):
        return self._scale / torch.sinh(torch.asinh(self._scale) * self.tailweight)
