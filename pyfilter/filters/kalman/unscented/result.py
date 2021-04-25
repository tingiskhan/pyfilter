import torch
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Module
from typing import Union, Tuple
from .utils import get_meancov
from ....timeseries import NewState


class UFTCorrectionResult(Module):
    def __init__(self, x: NewState, state_slice: slice, y: NewState):
        super().__init__()

        self.x = x
        self._state_slice = state_slice
        self.y = y

    def x_dist(self) -> Union[MultivariateNormal, Normal]:
        if self._state_slice.stop == 1:
            return Normal(self.x.dist.mean[..., 0], self.x.dist.scale_tril[..., 0, 0])

        return MultivariateNormal(
            self.x.dist.mean[..., self._state_slice],
            scale_tril=self.x.dist.scale_tril[..., self._state_slice, self._state_slice]
        )

    def calculate_sigma_points(self, cov_scale: float):
        choleskied_cov = cov_scale * self.x.dist.scale_tril

        mean = self.x.dist.mean

        spx = mean.unsqueeze(-2)
        sph = mean[..., None, :] + choleskied_cov
        spy = mean[..., None, :] - choleskied_cov

        return torch.cat((spx, sph, spy), -2)


class UFTPredictionResult(Module):
    def __init__(self, spx: NewState, spy: NewState):
        super().__init__()
        self.spx = spx
        self.spy = spy

    def get_mean_and_covariance(
            self, wm: torch.Tensor, wc: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        x_m, x_c = get_meancov(self.spx.values, wm, wc)
        y_m, y_c = get_meancov(self.spy.values, wm, wc)

        return (x_m, x_c), (y_m, y_c)
