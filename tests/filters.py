import numpy as np
import pytest
import torch
from pyfilter.timeseries import models as m, LinearGaussianObservations, AffineJointStochasticProcesses
from pyfilter.filters.particle import SISR, APF, proposals as props, ParticleFilter
from pyfilter.filters.kalman import UKF
from pykalman import KalmanFilter


@pytest.fixture
def linear_models():
    ar = m.AR(0.0, 0.99, 0.05)
    obs_1d_1d = LinearGaussianObservations(ar, 1.0, 0.15)

    kalman_1d_1d = KalmanFilter(
        transition_matrices=obs_1d_1d.hidden.parameter_1,
        observation_matrices=obs_1d_1d.observable.parameter_0,
        transition_covariance=obs_1d_1d.hidden.parameter_2 ** 2.0,
        transition_offsets=obs_1d_1d.hidden.parameter_0,
        observation_covariance=obs_1d_1d.observable.parameter_1 ** 2.0,
        initial_state_covariance=(obs_1d_1d.hidden.parameter_2 / (1 - obs_1d_1d.hidden.parameter_1)) ** 2.0
    )

    rw = m.RandomWalk(torch.tensor([0.05, 0.1]))
    obs_2d2_d = LinearGaussianObservations(rw, torch.eye(2), 0.15 * torch.ones(2))

    state_covariance = rw.parameter_0 ** 2.0 * np.eye(2)
    kalman_2d_2d = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=np.eye(2),
        transition_covariance=state_covariance,
        observation_covariance=obs_2d2_d.observable.parameter_1 ** 2.0 * np.eye(2),
        initial_state_covariance=state_covariance
    )

    llt = m.LocalLinearTrend(torch.tensor([0.01, 0.05]))
    obs_2d_1d = LinearGaussianObservations(llt, torch.tensor([0.0, 1.0]), 0.15)

    state_covariance_2 = llt.parameter_1.pow(2.0).cumsum(0) * np.eye(2)
    kalman_2d_1d = KalmanFilter(
        transition_matrices=llt.parameter_0,
        observation_matrices=obs_2d_1d.observable.parameter_0,
        transition_covariance=state_covariance_2,
        observation_covariance=obs_2d_1d.observable.parameter_1 ** 2.0,
        initial_state_covariance=state_covariance_2
    )

    # TODO: Add more models
    return (
        [obs_1d_1d, kalman_1d_1d],
        [obs_2d2_d, kalman_2d_2d],
        [obs_2d_1d, kalman_2d_1d],
    )


def construct_filters(model, **kwargs):
    particle_types = (SISR, APF)

    return (
        *(pt(model, 500, proposal=props.LinearGaussianObservations(), **kwargs) for pt in particle_types),
        UKF(model, **kwargs),
        *(pt(model, 5000, proposal=props.Bootstrap(), **kwargs) for pt in particle_types),
        *(pt(model, 500, proposal=props.Linearized(n_steps=5), **kwargs) for pt in particle_types),
        *(pt(model, 500, proposal=props.Linearized(n_steps=5, use_second_order=True), **kwargs) for pt in particle_types),
    )


@pytest.fixture
def sde():
    sde_ = m.Verhulst(0.01, 2.0, 0.05, dt=0.2, num_steps=5)

    return LinearGaussianObservations(sde_, 1.0, 0.05)


@pytest.fixture
def joint_timeseries():
    rw1 = m.RandomWalk(0.05)
    rw2 = m.RandomWalk(0.1)

    joint = AffineJointStochasticProcesses(rw1=rw1, rw2=rw2)

    obs = LinearGaussianObservations(joint, torch.eye(2), 0.15 * torch.ones(2))

    state_cov = np.array([0.05, 0.1]) ** 2.0 * np.eye(2)
    kalman = KalmanFilter(
        transition_matrices=np.eye(2),
        transition_covariance=state_cov,
        observation_matrices=np.eye(2),
        observation_covariance=obs.observable.parameter_1 ** 2.0 * np.eye(2),
        initial_state_covariance=state_cov
    )

    return obs, kalman


class TestFilters(object):
    RELATIVE_TOLERANCE = 1e-1
    PARALLEL_FILTERS = 20
    SERIES_LENGTH = 100
    PREDICTION_STEPS = 5
    STATE_RECORD_LENGTH = 5

    def test_compare_with_kalman_filter(self, linear_models):
        for model, kalman_model in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            kalman_mean, _ = kalman_model.filter(y.numpy())
            kalman_ll = kalman_model.loglikelihood(y.numpy())

            for f in construct_filters(model):
                result = f.longfilter(y)

                assert len(result.states) == 1
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

    def test_smoothing(self, linear_models):
        for model, kalman_model in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            kalman_mean, _ = kalman_model.smooth(y.numpy())

            for f in construct_filters(model, record_states=True):
                # Currently UKF does not implement smoothing
                if isinstance(f, UKF):
                    continue

                result = f.longfilter(y)
                smoothed_x = f.smooth(result.states)

                if model.hidden.n_dim < 1:
                    smoothed_x.unsqueeze_(-1)

                assert smoothed_x.shape[:2] == torch.Size([self.SERIES_LENGTH + 1, *f.particles])
                assert ((smoothed_x[1:].mean(1) - kalman_mean) / kalman_mean).abs().median() < self.RELATIVE_TOLERANCE

    def test_prediction(self, linear_models):
        for model, _ in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            for f in construct_filters(model):
                result = f.longfilter(y)

                x_pred, y_pred = f.predict(result.latest_state, self.PREDICTION_STEPS)

                assert x_pred.shape[:1] == y_pred.shape[:1] == torch.Size([self.PREDICTION_STEPS])

    def test_partial_state_history(self, linear_models):
        for model, _ in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            for f in construct_filters(model, record_states=self.STATE_RECORD_LENGTH):
                result = f.longfilter(y)

                assert (len(result.states) == self.STATE_RECORD_LENGTH) and (result.states[-1] is result.latest_state)

    def test_callback(self, linear_models):
        class Callback(object):
            def __init__(self):
                self.calls = 0

            def __call__(self, obj, x_):
                self.calls += 1

        for model, _ in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            cb = Callback()
            for f in construct_filters(model, pre_append_callbacks=[cb]):
                result = f.longfilter(y)

                assert cb.calls % self.SERIES_LENGTH == 0

    def test_sde(self, sde):
        x, y = sde.sample_path(self.SERIES_LENGTH)

        for f in construct_filters(sde):
            if not (isinstance(f, UKF) or (isinstance(f, ParticleFilter) and isinstance(f.proposal, props.Bootstrap))):
                continue

            result = f.longfilter(y)

            filter_std = result.filter_variance ** 0.5

            assert (filter_std <= sde.observable.parameter_1)[1:].all()

    def test_joint_timeseries(self, joint_timeseries):
        model, kalman = joint_timeseries

        x, y = model.sample_path(self.SERIES_LENGTH)
        kalman_mean, _ = kalman.filter(y.numpy())

        for f in construct_filters(model):
            result = f.longfilter(y)

            means = result.filter_means[1:]
            assert ((means - kalman_mean) / kalman_mean).abs().median() < self.RELATIVE_TOLERANCE
