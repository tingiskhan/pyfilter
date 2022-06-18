import random
import numpy as np
import pytest
import torch
from stochproc import timeseries as ts, distributions as dist
from pyro.distributions import Normal
from pykalman import KalmanFilter


def build_0d_dist(x, a, s):
    return Normal(loc=a * x.values, scale=s)


def build_1d_dist(x, a, s):
    return build_2d_to_1d_dist(x, a, s).to_event(1)


def build_2d_to_1d_dist(x, a, s):
    return Normal(loc=a.matmul(x.values.unsqueeze(-1)).squeeze(-1), scale=s)


def build_joint(x, a, s):
    return build_2d_to_1d_dist(x, a, s).to_event(1)


def linear_models():
    alpha, beta, sigma = 0.0, 0.99, 0.05
    ar = ts.models.AR(alpha, beta, sigma)

    a, s = 1.0, 0.15
    obs_1d_1d = ts.StateSpaceModel(ar, build_0d_dist, parameters=(a, s))

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

    inc_dist = dist.DistributionModule(Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1)

    rw = ts.LinearModel(
        torch.from_numpy(a).float(), torch.from_numpy(sigma).float(), increment_dist=inc_dist, initial_dist=inc_dist
    )

    params = torch.from_numpy(a).float(), torch.from_numpy(s).float()
    obs_2d_2d = ts.StateSpaceModel(rw, build_1d_dist, parameters=params)

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

    joint = ts.AffineJointStochasticProcess(proc_1=proc_1, proc_2=proc_2)

    eye = torch.eye(2)
    joint_ssm = ts.StateSpaceModel(joint, build_joint, (eye, s))

    state_covariance = sigma ** 2.0 * np.eye(2)
    kalman_2d_2d = KalmanFilter(
        transition_matrices=eye.numpy(),
        observation_matrices=eye.numpy(),
        transition_covariance=state_covariance,
        observation_covariance=s ** 2.0 * np.eye(2),
        initial_state_covariance=state_covariance
    )

    yield joint_ssm, kalman_2d_2d
