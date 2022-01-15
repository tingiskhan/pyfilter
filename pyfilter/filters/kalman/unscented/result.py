import torch
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Module
from typing import Union
from .utils import get_mean_and_cov, get_bad_inds, _EPS
from ....timeseries import NewState


class UFTCorrectionResult(Module):
    """
    State object for storing correction results of ``UnscentedFilterTransform``.
    """

    def __init__(self, x: NewState, state_slice: slice, y: NewState):
        """
        Initializes the ``UFTCorrectionResult`` class.

        Args:
            x: The state of the latent process together with the increment and observation density. See original paper
                for more details.
            state_slice: The slice which selects the state from ``x.values``.
            y: The state of the observation process.
        """

        super().__init__()

        self.x = x
        self._state_slice = state_slice
        self.y = y

    def x_dist(self) -> Union[MultivariateNormal, Normal]:
        """
        Returns the distribution of the latent process.
        """

        if self._state_slice.stop == 1:
            return Normal(self.x.dist.mean[..., 0], self.x.dist.scale_tril[..., 0, 0], validate_args=False)

        return MultivariateNormal(
            self.x.dist.mean[..., self._state_slice],
            scale_tril=self.x.dist.scale_tril[..., self._state_slice, self._state_slice],
            validate_args=False,
        )

    def calculate_sigma_points(self, cov_scale: float):
        """
        Calculates the sigma points from ``.dist.mean`` and ``cov_scale``.

        Args:
            cov_scale: Scaling factor for the choleskied covariance.

        Returns:
            Returns the sigma points required by ``UnscentedFilterTransform``.
        """

        choleskied_cov = cov_scale * self.x.dist.scale_tril

        mean = self.x.dist.mean

        spx = mean.unsqueeze(-2)
        sph = mean[..., None, :] + choleskied_cov
        spy = mean[..., None, :] - choleskied_cov

        return torch.cat((spx, sph, spy), -2)

    def resample(self, indices: torch.Tensor):
        """
        Resamples ``self``, only used when ``UnscentedFilterTransform`` is batched.

        Args:
            indices: The indices of the batches to select.
        """

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

    def exchange(self, indices: torch.Tensor, other: "UFTCorrectionResult"):
        """
        Exchanges the attributes of ``self`` with that of ``state`` at ``indices``. Only used when
        ``UnscentedFilterTransform`` is batched.

        Args:
            indices: See ``.resample(...)``.
            other: The other state to exchange with.
        """

        x_loc = self.x.dist.loc
        x_loc[indices] = other.x.dist.loc[indices]

        x_scale_tril = self.x.dist.scale_tril
        x_scale_tril[indices] = other.x.dist.scale_tril[indices]

        x_dist = MultivariateNormal(x_loc, scale_tril=x_scale_tril, validate_args=False)

        y_loc = self.y.dist.loc
        y_loc[indices] = other.y.dist.loc[indices]

        if isinstance(self.y.dist, Normal):
            y_scale = self.y.dist.scale
            y_scale[indices] = other.y.dist.scale[indices]

            y_dist = Normal(y_loc, y_scale, validate_args=False)
        else:
            y_scale = self.y.dist.scale_tril
            y_scale[indices] = other.y.dist.scale_tril[indices]

            y_dist = MultivariateNormal(y_loc, scale_tril=y_scale, validate_args=False)

        self.x = self.x.copy(dist=x_dist)
        self.y = self.y.copy(dist=y_dist)

    def update(
        self,
        xm: torch.Tensor,
        xc: torch.Tensor,
        x_state: NewState,
        prev_corr: "UFTCorrectionResult",
        ym: torch.Tensor = None,
        yc: torch.Tensor = None,
        y_state: NewState = None,
    ) -> "UFTCorrectionResult":
        """
        Creates a new state given the latest estimate of the means and covariances together with the previous correction
        state object.

        Args:
            xm: The estimated mean of the latent process.
            xc: The estimate covariance of the latent process.
            x_state: The latest state of the latent process, contains the sigma points.
            prev_corr: The previous correction state.
            ym: The estimated mean of the observation process.
            yc: THe estimate covariance of the observation process.
            y_state: The latest state of the observation process, contains the sigma points.

        Returns:
            A new ``UTFCorrectionResult`` object with updated means and covariances.
        """

        mean = prev_corr.x.dist.mean.clone()
        cov = prev_corr.x.dist.covariance_matrix.clone()

        # We only allow fixing bad matrices when we're running batched mode
        if cov.dim() > 2:
            bad_matrix = get_bad_inds(xc, yc)

            if bad_matrix.all():
                raise Exception("All batches were singular or nan!")

            xc[bad_matrix, self._state_slc, self._state_slc] = _EPS * torch.eye(xc.shape[-1], device=xc.device)
            xm[bad_matrix, self._state_slc] = 0.0

            yc[bad_matrix] = _EPS * torch.eye(yc.shape[-1], device=yc.device)
            ym[bad_matrix] = 0.0

        mean[..., self._state_slc] = xm
        cov[..., self._state_slc, self._state_slc] = xc

        x_state = x_state.copy(dist=MultivariateNormal(loc=mean, covariance_matrix=cov, validate_args=False))

        if self._model.observable.n_dim > 0:
            y_state = y_state.copy(dist=MultivariateNormal(loc=ym, covariance_matrix=yc, validate_args=False))
        else:
            y_state = y_state.copy(dist=Normal(loc=ym[..., 0], scale=yc[..., 0, 0].sqrt(), validate_args=False))

        return UFTCorrectionResult(x_state, self._state_slc, y_state)


class UFTPredictionResult(Module):
    """
    State object for storing prediction results of ``UnscentedFilterTransform``.
    """

    def __init__(self, spx: NewState, spy: NewState):
        """
        Initializes the ``UFTPredictionResult`` class.

        Args:
            spx: The sigma points of the latent process.
            spy: The sigma points of the observable process.
        """

        super().__init__()
        self.spx = spx
        self.spy = spy

    def get_mean_and_covariance(
        self, mean_weights: torch.Tensor, covariance_weights: torch.Tensor
    ) -> ((torch.Tensor, torch.Tensor), ...):
        """
        Calculates the means and covariances of the latent and observable processes from the sigma points of ``self``.

        Args:
            mean_weights: The weights to use for aggregating the means.
            covariance_weights: The weights to use for aggregating the covariances.

        Returns:
            Returns the tuple ``((latent mean, latent covariance), (observable mean, observable covariance))``.
        """

        x_m, x_c = get_mean_and_cov(self.spx.values, mean_weights, covariance_weights)
        y_m, y_c = get_mean_and_cov(self.spy.values, mean_weights, covariance_weights)

        return (x_m, x_c), (y_m, y_c)
