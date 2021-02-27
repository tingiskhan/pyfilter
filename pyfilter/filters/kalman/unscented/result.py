import torch
from torch.distributions import Normal, MultivariateNormal
from torch.nn import Module
from ....timeseries import TimeseriesState
from .utils import get_meancov


class UFTCorrectionResult(Module):
    def __init__(
        self, mean: TimeseriesState, cov: torch.Tensor, state_slice: slice, ym: torch.Tensor, yc: torch.Tensor
    ):
        super().__init__()
        self.register_buffer("ym", ym)
        self.register_buffer("yc", yc)

        self.add_module("mean", mean)
        self.register_buffer("cov", cov)
        self._state_slice = state_slice

    @property
    def xm(self):
        return self._modules["mean"].state[..., self._state_slice]

    @property
    def xc(self):
        return self._buffers["cov"][..., self._state_slice, self._state_slice]

    @staticmethod
    def _helper(m, c):
        if m.shape[-1] > 1:
            return MultivariateNormal(m, c)

        return Normal(m[..., 0], c[..., 0, 0].sqrt())

    def x_dist(self):
        return self._helper(self.xm, self.xc)

    def y_dist(self):
        return self._helper(self.ym, self.yc)

    def calculate_sigma_points(self, cov_scale: float):
        cholcov = cov_scale * torch.cholesky(self._buffers["cov"])

        spx = self._modules["mean"].state.unsqueeze(-2)
        sph = self._modules["mean"].state[..., None, :] + cholcov
        spy = self._modules["mean"].state[..., None, :] - cholcov

        return torch.cat((spx, sph, spy), -2)


class AggregatedResult(Module):
    def __init__(self, xm, xc, ym, yc):
        super().__init__()
        self.xm = xm
        self.xc = xc
        self.ym = ym
        self.yc = yc


class UFTPredictionResult(Module):
    def __init__(self, spx: TimeseriesState, spy: TimeseriesState):
        super().__init__()
        self.spx = spx
        self.spy = spy

    def get_mean_and_covariance(self, wm: torch.Tensor, wc: torch.Tensor) -> AggregatedResult:
        x_m, x_c = get_meancov(self.spx.state, wm, wc)
        y_m, y_c = get_meancov(self.spy.state, wm, wc)

        return AggregatedResult(x_m, x_c, y_m, y_c)
