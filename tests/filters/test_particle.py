import itertools
import pytest
import torch
from pyfilter.filters import particle as part
import numpy as np

from .models import linear_models


def construct_filters(particles=1_000, **kwargs):
    particle_types = (part.SISR, part.APF)

    for pt in particle_types:
        yield lambda m: pt(m, particles, proposal=part.proposals.Bootstrap(), **kwargs)
        yield lambda m: pt(m, particles, proposal=part.proposals.Linearized(n_steps=5), **kwargs)
        yield lambda m: pt(m, particles, proposal=part.proposals.Linearized(n_steps=5, use_second_order=True), **kwargs)

        linear_proposal = part.proposals.LinearGaussianObservations(parameter_index=0)
        yield lambda m: pt(m, particles, proposal=linear_proposal)


BATCH_SIZES = [
    torch.Size([]),
    torch.Size([10]),
]

MISSING_PERC = [0.0, 0.1]


def create_params():
    return itertools.product(linear_models(), construct_filters(), BATCH_SIZES, MISSING_PERC)


class TestParticleFilters(object):
    RELATIVE_TOLERANCE = 1e-1
    SERIES_LENGTH = 100

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

        low = means - std
        high = means + std

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
