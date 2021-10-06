import numpy as np
import pytest
import torch
from pyfilter.timeseries import models as m, LinearGaussianObservations
from pyfilter.filters.particle import SISR, APF, proposals as props
from pyfilter.filters.kalman import UKF
from pykalman import KalmanFilter


torch.manual_seed(123)


@pytest.fixture
def linear_models():
    ar = m.AR(0.0, 0.99, 0.05)
    obs_1d = LinearGaussianObservations(ar, 1.0, 0.15)

    kalman_1d = KalmanFilter(
        transition_matrices=obs_1d.hidden.parameter_1,
        observation_matrices=obs_1d.observable.parameter_0,
        transition_covariance=obs_1d.hidden.parameter_2 ** 2.0,
        transition_offsets=obs_1d.hidden.parameter_0,
        observation_covariance=obs_1d.observable.parameter_1 ** 2.0,
        initial_state_covariance=(obs_1d.hidden.parameter_2 / (1 - obs_1d.hidden.parameter_1)) ** 2.0
    )

    rw = m.RandomWalk(torch.tensor([0.05, 0.1]))
    obs_2d = LinearGaussianObservations(rw, torch.eye(2), 0.15 * torch.ones(2))

    state_covariance = rw.parameter_0 ** 2.0 * np.eye(2)
    kalman_2d = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=np.eye(2),
        transition_covariance=state_covariance,
        observation_covariance=obs_2d.observable.parameter_1 ** 2.0 * np.eye(2),
        initial_state_covariance=state_covariance
    )

    # TODO: Add more models
    return (
        [obs_1d, kalman_1d],
        [obs_2d, kalman_2d],
    )


def construct_filters(model):
    particle_types = (SISR, APF)

    return (
        *(pt(model, 500, proposal=props.LinearGaussianObservations()) for pt in particle_types),
        UKF(model),
        *(pt(model, 5000, proposal=props.Bootstrap()) for pt in particle_types),
        *(pt(model, 500, proposal=props.Linearized(n_steps=5)) for pt in particle_types),
        *(pt(model, 500, proposal=props.Linearized(n_steps=5, use_second_order=True)) for pt in particle_types),
    )


class TestFilters(object):
    RELATIVE_TOLERANCE = 1e-1
    PARALLEL_FILTERS = 20
    SERIES_LENGTH = 500

    def test_compare_with_kalman_filter(self, linear_models):
        for model, kalman_model in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            kalman_mean, _ = kalman_model.filter(y.numpy())
            kalman_ll = kalman_model.loglikelihood(y.numpy())

            for f in construct_filters(model):
                result = f.longfilter(y)

                assert ((result.loglikelihood - kalman_ll) / kalman_ll).abs() < self.RELATIVE_TOLERANCE

                means = result.filter_means[1:]
                if model.hidden.n_dim < 1:
                    means.unsqueeze_(-1)

                assert ((means - kalman_mean) / kalman_mean).abs().median() < self.RELATIVE_TOLERANCE

    def test_parallel_filters(self, linear_models):
        for model, kalman_model in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            kalman_mean, _ = kalman_model.filter(y.numpy())
            if model.hidden.n_dim > 0:
                kalman_mean = kalman_mean[:, None]

            kalman_ll = kalman_model.loglikelihood(y.numpy())

            for f in construct_filters(model):
                f.set_num_parallel(self.PARALLEL_FILTERS)

                result = f.longfilter(y)

                assert result.filter_means.shape[:2] == torch.Size([self.SERIES_LENGTH + 1, self.PARALLEL_FILTERS])
                assert ((result.loglikelihood - kalman_ll) / kalman_ll).abs().median() < self.RELATIVE_TOLERANCE

                means = result.filter_means[1:]
                assert ((means - kalman_mean) / kalman_mean).abs().median() < self.RELATIVE_TOLERANCE
