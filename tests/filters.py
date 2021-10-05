import pytest
from pyfilter.timeseries import models as m, LinearGaussianObservations
from pyfilter.filters.particle import SISR, APF, proposals as props
from pyfilter.filters.kalman import UKF
from pykalman import KalmanFilter
import numpy as np


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

    # TODO: Add more models
    return (
        [obs_1d, kalman_1d],
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
    RELATIVE_TOLERANCE = 5e-2

    def test_compare_with_kalman_filter(self, linear_models):
        for model, kalman_model in linear_models:
            x, y = model.sample_path(500)

            kalman_mean, _ = kalman_model.filter(y.numpy())
            kalman_ll = kalman_model.loglikelihood(y.numpy())

            for f in construct_filters(model):
                filter_result = f.longfilter(y)

                assert ((filter_result.loglikelihood - kalman_ll) / kalman_ll).abs() < self.RELATIVE_TOLERANCE

                means = filter_result.filter_means[1:]
                if model.hidden.n_dim < 1:
                    means.unsqueeze_(-1)

                assert ((means - kalman_mean) / kalman_mean).abs().median() < self.RELATIVE_TOLERANCE