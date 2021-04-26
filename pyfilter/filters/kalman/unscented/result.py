import torch
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Module
from typing import Union, Tuple
from .utils import get_meancov
from ....timeseries import NewState


class UFTCorrectionResult(Module):
    """
    Class for storing correction results of the Unscented Filter Transform.
    """

    def __init__(self, x: NewState, state_slice: slice, y: NewState):
        super().__init__()

        self.x = x
        self._state_slice = state_slice
        self.y = y

    def x_dist(self) -> Union[MultivariateNormal, Normal]:
        if self._state_slice.stop == 1:
            return Normal(self.x.dist.mean[..., 0], self.x.dist.scale_tril[..., 0, 0], validate_args=False)

        return MultivariateNormal(
            self.x.dist.mean[..., self._state_slice],
            scale_tril=self.x.dist.scale_tril[..., self._state_slice, self._state_slice],
            validate_args=False
        )

    def calculate_sigma_points(self, cov_scale: float):
        choleskied_cov = cov_scale * self.x.dist.scale_tril

        mean = self.x.dist.mean

        spx = mean.unsqueeze(-2)
        sph = mean[..., None, :] + choleskied_cov
        spy = mean[..., None, :] - choleskied_cov

        return torch.cat((spx, sph, spy), -2)

    def resample(self, indices: torch.Tensor):
        x_dist = MultivariateNormal(
            self.x.dist.loc[indices], scale_tril=self.x.dist.scale_tril[indices], validate_args=False
        )

        if isinstance(self.y.dist, Normal):
            y_dist = Normal(self.y.dist.loc[indices], self.y.dist.scale[indices], validate_args=False)
        else:
            y_dist = MultivariateNormal(
                self.y.dist.loc[indices], scale_tril=self.y.dist.scale_tril[indices], validate_args=False
            )

        self.x = self.x.copy(dist=x_dist)
        self.y = self.y.copy(dist=y_dist)

    def exchange(self, indices: torch.Tensor, state: "UFTCorrectionResult"):
        x_loc = self.x.dist.loc
        x_loc[indices] = state.x.dist.loc[indices]

        x_scale_tril = self.x.dist.scale_tril
        x_scale_tril[indices] = state.x.dist.scale_tril[indices]

        x_dist = MultivariateNormal(x_loc, scale_tril=x_scale_tril, validate_args=False)

        y_loc = self.y.dist.loc
        y_loc[indices] = state.y.dist.loc[indices]

        if isinstance(self.y.dist, Normal):
            y_scale = self.y.dist.scale
            y_scale[indices] = state.y.scale[indices]

            y_dist = Normal(y_loc, y_scale, validate_args=False)
        else:
            y_scale = self.y.dist.scale_tril
            y_scale[indices] = state.y.dist.scale_tril[indices]

            y_dist = MultivariateNormal(y_loc, scale_tril=y_scale, validate_args=False)

        self.x = self.x.copy(dist=x_dist)
        self.y = self.y.copy(dist=y_dist)


class UFTPredictionResult(Module):
    """
    Class for storing prediction results of the Unscented Filter Transform.
    """

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
