import torch.cuda
from stochproc import timeseries as ts
from pyro.distributions import Normal, LogNormal, Exponential
from pyfilter import inference as inf


def build_0d_dist(x, a, s):
    return Normal(loc=a * x.value, scale=s)


def build_obs_1d(model, a, s):
    return ts.StateSpaceModel(model, build_0d_dist, parameters=(a, s))


def linear_models():
    kappa = 0.025
    gamma = 0.0
    sigma = 0.05

    model = ts.models.OrnsteinUhlenbeck(kappa, gamma, sigma)

    yield build_obs_1d(model, 1.0, 0.05), build_model


def build_model(cntxt, use_cuda: bool = True):
    device = "cpu:0"
    if torch.cuda.is_available() and use_cuda:
        device = "cuda:0"

    kappa = cntxt.named_parameter("kappa", Exponential(rate=10.0).to(device))
    gamma = cntxt.named_parameter("gamma", Normal(loc=0.0, scale=1.0).to(device))
    sigma = cntxt.named_parameter("sigma", LogNormal(loc=-2.0, scale=1.0).to(device))

    prob_model = ts.models.OrnsteinUhlenbeck(kappa, gamma, sigma)

    return build_obs_1d(prob_model, torch.tensor(1.0, device=kappa.device), torch.tensor(0.05, device=kappa.device))
