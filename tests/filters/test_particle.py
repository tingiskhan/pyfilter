from functools import partial
import pytest
import torch
from pyfilter.filters import particle as part
import numpy as np

from .models import linear_models, local_linearization


def median_relative_deviation(y_true, y):
    return np.median(np.abs((y_true - y) / y_true))


def _create_partial(filter_class, particles, **kwargs):
    p = partial(filter_class, particles=particles, **kwargs)
    p.__repr__ = lambda u: f"{filter_class.__name__}({', '.join((f'{k}={v}' for k, v in kwargs.items()))})"

    return p


def construct_filters(particles=1_500, skip_gpf=False, **kwargs):
    if not skip_gpf:
        yield _create_partial(part.GPF, particles=particles, **kwargs)

        for use_second_order in [False, True]:
            yield _create_partial(part.GPF, particles=particles, proposal=part.proposals.GaussianLinearized(n_steps=5, use_second_order=use_second_order), **kwargs)

        yield _create_partial(part.GPF, particles=particles, proposal=part.proposals.GaussianLinear(), **kwargs)

    for pt in (part.APF, part.SISR):
        yield _create_partial(pt, particles=particles, proposal=part.proposals.Bootstrap(), **kwargs)
        yield _create_partial(pt, particles=particles, proposal=part.proposals.NestedProposal(50), **kwargs)

        for use_hessian in [False, True]:
            for use_functorch in [False, True]:
                proposal = part.proposals.Linearized(n_steps=5, use_second_order=use_hessian, use_functorch=use_functorch)
                yield _create_partial(pt, particles=particles, proposal=proposal, **kwargs)

        proposal = part.proposals.LinearGaussianObservations()
        yield _create_partial(pt, particles=particles, proposal=proposal, **kwargs)


def mask_missing(missing_percent: float, series_length: int, y, y_tensor):
    if missing_percent == 0.0:
        return 

    num_missing = int(missing_percent * series_length)
    indices = np.random.randint(1, y.shape[0], size=num_missing)

    y[indices] = np.ma.masked
    y_tensor[indices] = float("nan")


BATCH_SIZES = [
    torch.Size([]),
    torch.Size([3]),
]

MISSING_PERC = [0.0, 0.1]


# TODO: Clean this up
class TestParticleFilters(object):
    RELATIVE_TOLERANCE = 1e-1
    SERIES_LENGTH = 100

    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("filter_", construct_filters())
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("missing_perc", MISSING_PERC)
    @pytest.mark.parametrize("test_copy", [False, True])
    def test_filter_and_log_likelihood(self, models, filter_, batch_size, missing_perc, test_copy):
        np.random.seed(123)
        torch.manual_seed(123)

        model, kalman_model = models
        x, y = kalman_model.sample(self.SERIES_LENGTH)
        y_tensor = torch.from_numpy(y).float()

        mask_missing(missing_perc, self.SERIES_LENGTH, y, y_tensor)

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
                assert (new_state.timeseries_state.value == copy_state.timeseries_state.value).all() 
                assert (new_state.normalized_weights() == copy_state.normalized_weights()).all()

        assert len(result.states) == 1
        assert (median_relative_deviation(kalman_ll, result.loglikelihood) < self.RELATIVE_TOLERANCE).all()

        means = result.filter_means[1:]

        deviation = median_relative_deviation(kalman_mean, means.cpu().numpy())

        assert deviation < self.RELATIVE_TOLERANCE

    @pytest.mark.parametrize("models", linear_models())
    @pytest.mark.parametrize("filter_", construct_filters())
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("missing_perc", MISSING_PERC)
    def test_predict(self, models, filter_, batch_size, missing_perc):
        model, kalman_model = models
        x, y = kalman_model.sample(self.SERIES_LENGTH)

        y_tensor = torch.from_numpy(y).float()
        mask_missing(missing_perc, self.SERIES_LENGTH, y, y_tensor)

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
        mask_missing(missing_perc, self.SERIES_LENGTH, y, y_tensor)

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
    @pytest.mark.parametrize("filter_", construct_filters(particles=1_500, record_states=True, skip_gpf=True))
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("missing_perc", MISSING_PERC)
    @pytest.mark.parametrize("method", ["ffbs", "fl"])
    def test_smooth(self, models, filter_, batch_size, missing_perc, method):
        np.random.seed(123)
        torch.manual_seed(123)

        model, kalman_model = models
        x, y = kalman_model.sample(self.SERIES_LENGTH)
        y_tensor = torch.from_numpy(y).float()

        mask_missing(missing_perc, self.SERIES_LENGTH, y, y_tensor)

        kalman_mean, _ = kalman_model.smooth(y)

        if len(batch_size) > 0:
            kalman_mean = kalman_mean[:, None]

        f: part.ParticleFilter = filter_(model)
        f.set_batch_shape(batch_size)

        result = f.batch_filter(y_tensor)
        assert len(result.states) == kalman_mean.shape[0] + 1

        smoothed = f.smooth(result.states, method=method)

        means = smoothed[1:].mean(1)

        if model.hidden.n_dim == 0:
            means.unsqueeze_(-1)

        means = means.cpu().numpy()

        if method != "fl":
            assert median_relative_deviation(kalman_mean[-int(0.9 * self.SERIES_LENGTH):], means[-int(0.9 * self.SERIES_LENGTH):]) < self.RELATIVE_TOLERANCE
        else:
            assert median_relative_deviation(kalman_mean[-10:], means[-10:]) < self.RELATIVE_TOLERANCE
