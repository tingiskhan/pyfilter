import random
import numpy as np
import pytest
import torch
import pyfilter.timeseries as ts
from pyfilter.filters import particle as part, kalman
from pykalman import KalmanFilter


@pytest.fixture
def linear_models():
    alpha, beta, sigma = 0.0, 0.99, 0.05
    a, s = 1.0, 0.15

    ar = ts.models.AR(alpha, beta, sigma)
    obs_1d_1d = ts.LinearGaussianObservations(ar, a, s)

    kalman_1d_1d = KalmanFilter(
        transition_matrices=beta,
        observation_matrices=a,
        transition_covariance=sigma ** 2.0,
        transition_offsets=alpha,
        observation_covariance=s ** 2.0,
        initial_state_mean=alpha,
        initial_state_covariance=sigma ** 2 / (1 - beta ** 2.0)
    )

    sigma = np.array([0.05, 0.1])
    a, s = np.eye(2), 0.15 * np.ones(2)

    rw = ts.models.RandomWalk(torch.from_numpy(sigma).float())
    obs_2d_2d = ts.LinearGaussianObservations(rw, torch.from_numpy(a).float(), torch.from_numpy(s).float())

    state_covariance = sigma ** 2.0 * np.eye(2)
    kalman_2d_2d = KalmanFilter(
        transition_matrices=a,
        observation_matrices=a,
        transition_covariance=state_covariance,
        observation_covariance=s ** 2.0 * np.eye(2),
    )

    sigma = np.array([0.005, 0.02])
    a, s = np.array([0.0, 1.0]), 0.15

    llt = ts.models.LocalLinearTrend(torch.from_numpy(sigma).float(), initial_scale=sigma)
    obs_2d_1d = ts.LinearGaussianObservations(llt, torch.from_numpy(a).float(), s)

    state_covariance_2 = sigma ** 2.0 * np.eye(2)
    kalman_2d_1d = KalmanFilter(
        transition_matrices=llt.parameters_and_buffers()["parameter_0"].numpy(),
        observation_matrices=a,
        transition_covariance=state_covariance_2,
        observation_covariance=s ** 2.0,
        initial_state_covariance=state_covariance_2
    )

    # TODO: Add more models
    return (
        [obs_1d_1d, kalman_1d_1d],
        [obs_2d_2d, kalman_2d_2d],
        [obs_2d_1d, kalman_2d_1d],
    )


def construct_filters(model, **kwargs):
    particle_types = (part.SISR, part.APF)

    if not isinstance(model.hidden, ts.models.LocalLinearTrend):
        yield kalman.UKF(model, **kwargs)

    for pt in particle_types:
        yield pt(model, 5000, proposal=part.proposals.Bootstrap(), **kwargs)
        yield pt(model, 500, proposal=part.proposals.Linearized(n_steps=5), **kwargs)
        yield pt(model, 500, proposal=part.proposals.Linearized(n_steps=5, use_second_order=True), **kwargs)

        if isinstance(model, ts.LinearGaussianObservations):
            yield pt(model, 500, proposal=part.proposals.LinearGaussianObservations(), **kwargs)


@pytest.fixture
def sde():
    sde_ = ts.models.Verhulst(0.01, 2.0, 0.05, dt=0.2, num_steps=5)

    return ts.LinearGaussianObservations(sde_, 1.0, 0.05)


@pytest.fixture
def joint_timeseries():
    rw1 = ts.models.RandomWalk(0.05)
    rw2 = ts.models.RandomWalk(0.1)

    joint = ts.AffineJointStochasticProcess(rw1=rw1, rw2=rw2)

    obs = ts.LinearGaussianObservations(joint, torch.eye(2), 0.05 * torch.ones(2))

    param_1 = rw1.parameters_and_buffers()["parameter_0"]
    param_2 = rw2.parameters_and_buffers()["parameter_0"]

    state_cov = torch.stack((param_1, param_2), dim=0) ** 2.0 * np.eye(2)
    kalman = KalmanFilter(
        transition_matrices=np.eye(2),
        transition_covariance=state_cov,
        observation_matrices=np.eye(2),
        observation_covariance=obs.observable.parameters_and_buffers()["parameter_1"] ** 2.0 * np.eye(2),
        initial_state_covariance=state_cov
    )

    return obs, kalman


def add_nan(y, number_of_nans):
    rand_ints = torch.randint(0, y.shape[0], size=(number_of_nans,))

    if y.dim() == 1:
        y[rand_ints] = float("nan")
    else:
        rand_col = random.randint(0, y.shape[-1] - 1)
        y[rand_ints, rand_col] = float("nan")

    return y


class TestFilters(object):
    RELATIVE_TOLERANCE = 1e-1
    PARALLEL_FILTERS = 20
    SERIES_LENGTH = 100
    PREDICTION_STEPS = 5
    STATE_RECORD_LENGTH = 5
    NUMBER_OF_NANS = 10

    def _compare_kalman_mean(self, kalman_mean, means):
        assert ((means - kalman_mean) / kalman_mean).abs().median() < self.RELATIVE_TOLERANCE

    def test_compare_with_kalman_filter(self, linear_models):
        for model, kalman_model in linear_models:
            x, y = kalman_model.sample(self.SERIES_LENGTH)

            kalman_mean, _ = kalman_model.filter(y)
            kalman_ll = kalman_model.loglikelihood(y)

            for f in construct_filters(model):
                result = f.longfilter(torch.from_numpy(y).float())

                assert len(result.states) == 1
                assert ((result.loglikelihood - kalman_ll) / kalman_ll).abs() < self.RELATIVE_TOLERANCE

                means = result.filter_means[1:]
                if model.hidden.n_dim < 1:
                    means.unsqueeze_(-1)

                self._compare_kalman_mean(kalman_mean, means)

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

                self._compare_kalman_mean(kalman_mean, result.filter_means[1:])

    def test_smoothing(self, linear_models):
        for model, kalman_model in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            kalman_mean, _ = kalman_model.smooth(y.numpy())

            for f in construct_filters(model, record_states=True):
                # Currently UKF does not implement smoothing
                if isinstance(f, kalman.UKF):
                    continue

                result = f.longfilter(y)
                smoothed_x = f.smooth(result.states)

                if model.hidden.n_dim < 1:
                    smoothed_x.unsqueeze_(-1)

                assert smoothed_x.shape[:2] == torch.Size([self.SERIES_LENGTH + 1, *f.particles])
                self._compare_kalman_mean(kalman_mean, smoothed_x[1:].mean(1))

    def test_prediction(self, linear_models):
        for model, _ in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            for f in construct_filters(model):
                result = f.longfilter(y)

                x_pred, y_pred = f.predict_path(result.latest_state, self.PREDICTION_STEPS)

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
            if not (isinstance(f, kalman.UKF) or (isinstance(f, part.ParticleFilter) and isinstance(f.proposal, part.proposals.Bootstrap))):
                continue

            result = f.longfilter(y)

            filter_std = result.filter_variance ** 0.5

            assert (filter_std <= sde.observable.parameters_and_buffers()["parameter_2"] * 1.2)[1:].float().mean() > 0.95

    def test_joint_timeseries(self, joint_timeseries):
        model, kalman = joint_timeseries

        x, y = model.sample_path(self.SERIES_LENGTH)
        kalman_mean, _ = kalman.filter(y.numpy())

        for f in construct_filters(model):
            result = f.longfilter(y)

            means = result.filter_means[1:]
            assert ((means - kalman_mean) / kalman_mean).abs().median() < self.RELATIVE_TOLERANCE

    def test_partial_moment_history(self, linear_models):
        for model, _ in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)

            for f in construct_filters(model, record_moments=self.STATE_RECORD_LENGTH):
                result = f.longfilter(y)

                assert len(result.filter_means) == self.STATE_RECORD_LENGTH

    def test_exogenous_variables(self):
        ar = ts.models.AR(0.0, 0.99, 0.05)

        def _f(x, sigma):
            return x.exog + x.values

        def _g(x, sigma):
            return sigma

        line = torch.arange(self.SERIES_LENGTH)
        obs = ts.AffineObservations((_f, _g), (0.05,), ar.increment_dist, exog=line)

        model = ts.StateSpaceModel(ar, obs)
        x, y = model.sample_path(self.SERIES_LENGTH)

        params = ar.parameters_and_buffers()
        kf = KalmanFilter(
            transition_matrices=params["parameter_0"],
            transition_covariance=params["parameter_2"] ** 2,
            observation_matrices=1.0,
            observation_offsets=line.unsqueeze(-1).numpy(),
            observation_covariance=obs.parameters_and_buffers()["parameter_0"] ** 2,
            initial_state_covariance=params["parameter_2"] ** 2 / (1 - params["parameter_0"] ** 2)
        )

        kalman_mean, _ = kf.filter(y.numpy())

        for f in construct_filters(model):
            result = f.longfilter(y)

            means = result.filter_means[1:]
            if model.hidden.n_dim < 1:
                means.unsqueeze_(-1)

            self._compare_kalman_mean(kalman_mean, means)

    def test_missing_data(self, linear_models):
        for model, _ in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)
            y = add_nan(y, self.NUMBER_OF_NANS)

            for strat in ["impute", "skip"]:
                for parallel in [0, 10]:
                    for f in construct_filters(model, nan_strategy=strat):
                        if parallel > 0:
                            f.set_num_parallel(parallel)

                        res = f.longfilter(y)

                        assert not torch.isnan(res.filter_means).any()

    def test_missing_data_smooth(self, linear_models):
        for model, kalman_model in linear_models:
            x, y = model.sample_path(self.SERIES_LENGTH)
            y = add_nan(y, self.NUMBER_OF_NANS)

            for f in construct_filters(model, record_states=True):
                # Currently UKF does not implement smoothing
                if isinstance(f, kalman.UKF):
                    continue

                result = f.longfilter(y)
                smoothed_x = f.smooth(result.states)

                assert not smoothed_x.isnan().any()
