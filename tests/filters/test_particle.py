import itertools
import pytest
import torch
from pyfilter.filters import particle as part
import numpy as np

from .models import linear_models, local_linearization


def median_relative_deviation(y_true, y):
    return np.median(np.abs((y_true - y) / y_true))


def construct_filters(particles=3_000, **kwargs):
    particle_types = (part.SISR, part.APF)

    for pt in particle_types:
        yield lambda m: pt(m, particles, proposal=part.proposals.Bootstrap(), **kwargs)
        yield lambda m: pt(m, particles, proposal=part.proposals.Linearized(n_steps=5), **kwargs)
        yield lambda m: pt(m, particles, proposal=part.proposals.Linearized(n_steps=5, use_second_order=True), **kwargs)

        linear_proposal = part.proposals.LinearGaussianObservations(a_index=0)
        yield lambda m: pt(m, particles, proposal=linear_proposal, **kwargs)


BATCH_SIZES = [
    torch.Size([]),
    torch.Size([10]),
]

MISSING_PERC = [0.0, 0.1]


class TestParticleFilters(object):
    RELATIVE_TOLERANCE = 1e-1
    SERIES_LENGTH = 100
    NUM_STDS = 3.0

    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("filter_", construct_filters())
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("missing_perc", MISSING_PERC)
    @pytest.mark.parametrize("test_copy", [False, True])
    def test_filter_and_log_likelihood(self, models, filter_, batch_size, missing_perc, test_copy):
        np.random.seed(123)

        model, kalman_model = models
        x, y = kalman_model.sample(self.SERIES_LENGTH)
        y_tensor = torch.from_numpy(y).float()

        if missing_perc > 0.0:
            num_missing = int(missing_perc * self.SERIES_LENGTH)
            indices = np.random.randint(1, y.shape[0], size=num_missing)

            y[indices] = np.ma.masked
            y_tensor[indices] = float("nan")

        kalman_mean, _ = kalman_model.filter(y)

        if len(batch_size) > 0:
            kalman_mean = kalman_mean[:, None]

        kalman_ll = kalman_model.loglikelihood(y)

        f: part.ParticleFilter = filter_(model)
        f.set_batch_shape(batch_size)
        result = f.batch_filter(y_tensor)

        if test_copy:
            old_result = result
            result = result.copy()

            assert (result is not old_result)

            for new_state, copy_state in zip(result.states, old_result.states):
                assert new_state is not copy_state
                assert (new_state.x.value == copy_state.x.value).all() 
                assert (new_state.normalized_weights() == copy_state.normalized_weights()).all()

        assert len(result.states) == 1
        assert (((result.loglikelihood - kalman_ll) / kalman_ll).abs() < self.RELATIVE_TOLERANCE).all()

        means = result.filter_means[1:]

        if model.hidden.n_dim == 0:
            means.unsqueeze_(-1)

        deviation = median_relative_deviation(kalman_mean, means.cpu().numpy())
        thresh = 1e-1

        assert deviation < thresh

    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("filter_", construct_filters())
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("missing_perc", MISSING_PERC)
    def test_predict(self, models, filter_, batch_size, missing_perc):
        model, kalman_model = models
        x, y = kalman_model.sample(self.SERIES_LENGTH)

        y_tensor = torch.from_numpy(y).float()
        if missing_perc > 0.0:
            num_missing = int(missing_perc * self.SERIES_LENGTH)
            indices = np.random.randint(1, y.shape[0], size=num_missing)

            y_tensor[indices] = float("nan")

        f: part.ParticleFilter = filter_(model)
        f.set_batch_shape(batch_size)
        result = f.batch_filter(y_tensor)

        num_steps = 10
        path = result.latest_state.predict_path(model, num_steps)

        assert len(path.get_paths()) == 2

        x, y = path.get_paths()

        assert x.shape == torch.Size([num_steps, *f.particles, *f.ssm.hidden.event_shape])

    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("filter_", construct_filters())
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("missing_perc", MISSING_PERC)
    def test_save_and_load(self, models, filter_, batch_size, missing_perc):
        model, kalman_model = models
        x, y = kalman_model.sample(self.SERIES_LENGTH)

        y_tensor = torch.from_numpy(y).float()
        if missing_perc > 0.0:
            num_missing = int(missing_perc * self.SERIES_LENGTH)
            indices = np.random.randint(1, y.shape[0], size=num_missing)

            y_tensor[indices] = float("nan")

        f: part.ParticleFilter = filter_(model)

        f.set_batch_shape(batch_size)
        result = f.batch_filter(y_tensor)

        state_dict = result.state_dict()

        new_result = f.initialize_with_result()
        new_result.load_state_dict(state_dict)

        assert (
                (new_result.filter_means == result.filter_means).all() and
                (new_result.filter_variance == result.filter_variance).all() and
                (new_result.loglikelihood == result.loglikelihood).all()
        )

        for new_s, old_s in zip(new_result.states, result.states):
            new_ts = new_s.get_timeseries_state()
            old_ts = old_s.get_timeseries_state()

            assert (new_ts.value == old_ts.value).all() and (new_ts.time_index == old_ts.time_index).all()

    # TODO: Use same method as for filter rather than copy paste
    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("filter_", construct_filters(particles=400, record_states=True))
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("missing_perc", MISSING_PERC)
    @pytest.mark.parametrize("method", ["ffbs", "fl"])
    def test_smooth(self, models, filter_, batch_size, missing_perc, method):
        np.random.seed(123)

        model, kalman_model = models
        x, y = kalman_model.sample(self.SERIES_LENGTH)
        y_tensor = torch.from_numpy(y).float()

        if missing_perc > 0.0:
            num_missing = int(missing_perc * self.SERIES_LENGTH)
            indices = np.random.randint(1, y.shape[0], size=num_missing)

            y[indices] = np.ma.masked
            y_tensor[indices] = float("nan")

        kalman_mean, _ = kalman_model.smooth(y)

        if len(batch_size) > 0:
            kalman_mean = kalman_mean[:, None]

        f: part.ParticleFilter = filter_(model)
        f.set_batch_shape(batch_size)

        result = f.batch_filter(y_tensor)
        assert len(result.states) == kalman_mean.shape[0] + 1

        smoothed = f.smooth(result.states, method=method)

        means = smoothed[1:].mean(dim=len(batch_size) + 1)

        if model.hidden.n_dim == 0:
            means.unsqueeze_(-1)

        means = means.cpu().numpy()

        thresh = 1e-1
        if method != "fl":
            assert median_relative_deviation(kalman_mean, means) < thresh
        else:
            assert median_relative_deviation(kalman_mean[-10:], means[-10:]) < thresh

    @pytest.mark.parametrize("batch_shape", BATCH_SIZES)
    @pytest.mark.parametrize("linearization", local_linearization())
    def test_local_linearization(self, batch_shape, linearization):
        model, (f, f_prime) = linearization

        x, y = model.sample_states(self.SERIES_LENGTH).get_paths()

        for filt in (part.SISR, part.APF):
            linearized_proposal = filt(model, 1_000, proposal=part.proposals.LocalLinearization(f, f_prime))
            linearized_proposal.set_batch_shape(batch_shape)
            linearized_result = linearized_proposal.batch_filter(y)

            mean = linearized_result.filter_means[1:]
            std = linearized_result.filter_variance[1:].sqrt()

            low = mean - 2 * std
            high = mean + 2 * std

            x_ = x.clone() if batch_shape.numel() == 1 else x.unsqueeze(1)

            # NB: Blunt, but kinda works...
            assert (((low <= x_) & (x_ <= high)).float().mean() > 0.75).all()
