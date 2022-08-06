import itertools
import pytest
import torch
from pyfilter.filters import particle as part
import numpy as np

from .models import linear_models, local_linearization


def construct_filters(particles=1_000, **kwargs):
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


def create_params(**kwargs):
    return itertools.product(linear_models(), construct_filters(**kwargs), BATCH_SIZES, MISSING_PERC)


class TestParticleFilters(object):
    RELATIVE_TOLERANCE = 1e-1
    SERIES_LENGTH = 100
    NUM_STDS = 3.0

    @pytest.mark.parametrize("models, filter_, batch_size, missing_perc", create_params())
    def test_compare_with_kalman_filter(self, models, filter_, batch_size, missing_perc):
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

        assert len(result.states) == 1
        assert (((result.loglikelihood - kalman_ll) / kalman_ll).abs() < self.RELATIVE_TOLERANCE).all()

        means = result.filter_means[1:]
        std = result.filter_variance[1:].sqrt()

        low = means - self.NUM_STDS * std
        high = means + self.NUM_STDS * std

        if model.hidden.n_dim < 1:
            low.unsqueeze_(-1)
            high.unsqueeze_(-1)

        low = low.numpy()
        high = high.numpy()

        assert ((low <= kalman_mean) & (kalman_mean <= high)).all()

    @pytest.mark.parametrize("models, filter_, batch_size, missing_perc", create_params())
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

    @pytest.mark.parametrize("models, filter_, batch_size, missing_perc", create_params())
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

            assert (new_ts.values == old_ts.values).all() and (new_ts.time_index == old_ts.time_index).all()

    @pytest.mark.parametrize("models, filter_", itertools.product(linear_models(), construct_filters()))
    def test_check_inactive_context_raises(self, models, filter_):
        model, _ = models

        from pyfilter.inference import make_context

        context = make_context()

        def model_builder(context_):
            return model

        # with pytest.raises()
        with pytest.raises(Exception):
            f = filter_(model_builder)

    # TODO: Use same method as for filter rather than copy paste
    @pytest.mark.parametrize("models, filter_, batch_size, missing_perc", create_params(particles=400, record_states=True))
    def test_smooth(self, models, filter_, batch_size, missing_perc):
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

        smoothed = f.smooth(result.states)

        means = smoothed[1:].mean(dim=len(batch_size) + 1)
        std = smoothed[1:].std(dim=len(batch_size) + 1)

        low = means - self.NUM_STDS * std
        high = means + self.NUM_STDS * std

        if model.hidden.n_dim < 1:
            low.unsqueeze_(-1)
            high.unsqueeze_(-1)

        low = low.numpy()
        high = high.numpy()

        assert ((low <= kalman_mean) & (kalman_mean <= high)).all()

    @pytest.mark.parametrize("batch_shape, linearization", itertools.product(BATCH_SIZES, local_linearization()))
    def test_local_linearization(self, batch_shape, linearization):
        model, (f, f_prime) = linearization

        x, y = model.sample_states(self.SERIES_LENGTH).get_paths()

        for filt in (part.SISR, part.APF):
            linearized_proposal = filt(model, 1_000, proposal=part.proposals.LocalLinearization(f, f_prime))
            linearized_proposal.set_batch_shape(batch_shape)
            linearized_result = linearized_proposal.batch_filter(y)

            mean = linearized_result.filter_means[1:]
            std = linearized_result.filter_variance[1:].sqrt()

            low = mean - self.NUM_STDS * std
            high = mean + self.NUM_STDS * std

            x_ = x.clone() if batch_shape.numel() == 1 else x.unsqueeze(1)

            # NB: Blunt, but kinda works...
            assert (((low <= x_) & (x_ <= high)).float().mean() > 0.75).all()
