import pytest
from pyfilter.timeseries import models as m, LinearGaussianObservations
from pyfilter.filters.particle import SISR, APF, proposals as props
from pyfilter.filters.kalman import UKF


@pytest.fixture
def linear_models():
    ar = m.AR(0.0, 0.99, 0.05)
    obs_1d = LinearGaussianObservations(ar, 1.0, 0.05)

    # TODO: Add more models
    return (obs_1d,)


@pytest.fixture
def filters(linear_models):
    filters_ = tuple()
    particle_types = (SISR, APF)

    for mod in linear_models:
        filters_ += (
            *(pt(mod, 500) for pt in particle_types),
            UKF(mod),
            *(pt(mod, 500, proposal=props.Bootstrap()) for pt in particle_types),
            *(pt(mod, 500, proposal=props.Linearized()) for pt in particle_types),
            *(pt(mod, 500, proposal=props.LocalLinearization()) for pt in particle_types),
        )

    return filters_


class TestFilters(object):
    def test_compare_with_kalman_filter(self, filters):
        for f in filters:
            x, y = f.ssm.sample_path(500)

            filter_result = f.longfilter(y, bar=False)
            # TODO: Compare with Kalman