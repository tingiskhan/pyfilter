from math import sqrt

import numpy as np
import torch
from stochproc import timeseries as ts, distributions as dist
from pyro.distributions import Normal
from pykalman import KalmanFilter


def build_0d_dist(x, a, s):
    return Normal(loc=a * x.value, scale=s)


def build_1d_dist(x, a, s):
    return build_2d_to_1d_dist(x, a, s).to_event(1)


def build_2d_to_1d_dist(x, a, s):
    return Normal(loc=a.matmul(x.value.unsqueeze(-1)).squeeze(-1), scale=s)


def build_joint(x, a, s):
    return build_2d_to_1d_dist(x, a, s).to_event(1)


def linear_models():
    alpha, beta, sigma = 0.0, 0.99, 0.05
    ar = ts.models.AR(alpha, beta, sigma)

    a, s = 1.0, 0.15
    obs_1d_1d = ts.LinearStateSpaceModel(ar, (a, s), torch.Size([]))

    kalman_1d_1d = KalmanFilter(
        transition_matrices=beta,
        observation_matrices=a,
        transition_covariance=sigma ** 2.0,
        transition_offsets=alpha,
        observation_covariance=s ** 2.0,
        initial_state_mean=alpha,
        initial_state_covariance=sigma ** 2.0
    )

    yield obs_1d_1d, kalman_1d_1d

    sigma = np.array([0.05, 0.1])
    a, s = np.eye(2), 0.15 * np.ones(2)

    inc_dist = Normal(loc=0.0, scale=1.0).expand(torch.Size([2])).to_event(1)

    parameters = (torch.from_numpy(a).float(), torch.from_numpy(sigma).float())
    rw = ts.LinearModel(
        parameters, inc_dist, lambda *args: Normal(0.0, 1.0).expand(torch.Size([2])).to_event(1)
    )

    params = torch.from_numpy(a).float(), torch.from_numpy(s).float()
    obs_2d_2d = ts.LinearStateSpaceModel(rw, params, torch.Size([2]))

    state_covariance = sigma ** 2.0 * np.eye(2)
    kalman_2d_2d = KalmanFilter(
        transition_matrices=a,
        observation_matrices=a,
        transition_covariance=state_covariance,
        observation_covariance=s ** 2.0 * np.eye(2),
    )

    yield obs_2d_2d, kalman_2d_2d

    sigma = 0.05
    proc_1 = ts.models.RandomWalk(sigma)
    proc_2 = ts.models.RandomWalk(sigma)

    joint = ts.joint_process(proc_1=proc_1, proc_2=proc_2)

    eye = torch.eye(2)
    joint_ssm = ts.LinearStateSpaceModel(joint, (eye, params[-1]), torch.Size([2]))

    state_covariance = sigma ** 2.0 * np.eye(2)
    kalman_2d_2d = KalmanFilter(
        transition_matrices=eye.numpy(),
        observation_matrices=eye.numpy(),
        transition_covariance=state_covariance,
        observation_covariance=s ** 2.0 * np.eye(2),
        initial_state_covariance=state_covariance
    )

    yield joint_ssm, kalman_2d_2d


def build_non_linear_mean(x, s):
    return x.value ** 2.0 / 20.0


def build_non_linear_deriv(x, s):
    return x.value / 10.0


def build_non_linear_dist(x, s):
    return Normal(loc=build_non_linear_mean(x, s), scale=s)


def mean_scale(x, s):
    x_t = x.value
    return x_t / 2.0 + 25 * x_t / (1 + x_t ** 2.0) + 8.0 * (1.2 * x.time_index).cos(), s


class UKFState(object):
    def __init__(self):
        self.t = 0.0

    def __call__(self, x_t):
        self.t += 1.0
        return x_t / 2.0 + 25 * x_t / (1 + x_t ** 2.0) + 8.0 * np.cos(1.2 * self.t)


def local_linearization():
    sigma = sqrt(10.0)
    
    inc_dist = Normal(loc=0.0, scale=1.0)
    ar = ts.AffineProcess(mean_scale, (sigma,), inc_dist, lambda *args: Normal(loc=0.0, scale=sqrt(5.0)))

    s = 1.0
    obs_1d_1d = ts.StateSpaceModel(ar, build_non_linear_dist, parameters=(s,))

    yield obs_1d_1d, (build_non_linear_mean, build_non_linear_deriv)
