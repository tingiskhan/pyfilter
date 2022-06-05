from stochproc import timeseries as ts
from pyro.distributions import Normal, LogNormal, Exponential
from pyfilter import inference as inf


def build_0d_dist(x, a, s):
    return Normal(loc=a * x.values, scale=s)


def build_obs_1d(model):
    a, s = 1.0, 0.075
    return ts.StateSpaceModel(model, build_0d_dist, parameters=(a, s))


def linear_models():
    kappa = 0.01
    gamma = 0.0
    sigma = 0.05

    model = ts.models.OrnsteinUhlenbeck(kappa, gamma, sigma)

    yield build_obs_1d(model), build_model


def build_model(cntxt):
    kappa = cntxt.named_parameter("kappa", inf.Prior(Exponential, rate=1.0))
    gamma = cntxt.named_parameter("gamma", inf.Prior(Normal, loc=0.0, scale=1.0))
    sigma = cntxt.named_parameter("sigma", inf.Prior(LogNormal, loc=-2.0, scale=0.5))

    prob_model = ts.models.OrnsteinUhlenbeck(kappa, gamma, sigma)

    return build_obs_1d(prob_model)
