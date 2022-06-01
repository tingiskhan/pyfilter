from stochproc import timeseries as ts
from pyro.distributions import Normal


def build_0d_dist(x, a, s):
    return Normal(loc=a * x.values, scale=s)


def build_obs_1d(model):
    a, s = 1.0, 0.15
    return ts.StateSpaceModel(model, build_0d_dist, parameters=(a, s))


def linear_models():
    sigma = 0.05
    ar = ts.models.RandomWalk(scale=sigma)

    yield build_obs_1d(ar)
