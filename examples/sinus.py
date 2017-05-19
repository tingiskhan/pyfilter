from pyfilter.model import StateSpaceModel
from pyfilter.timeseries.meta import Base
from pyfilter.timeseries.observable import Observable
from pyfilter.filters.rapf import RAPF
from pyfilter.distributions.continuous import Gamma, Normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def fh0(alpha, sigma):
    return 0


def gh0(alpha, sigma):
    return sigma


def fh(x, alpha, sigma):
    return x + np.sin(x - alpha)


def gh(x, alpha, sigma):
    return sigma


def go(x, beta):
    return x


def fo(x, beta):
    return beta

# ===== SIMULATE SSM ===== #

sinus = Base((fh0, gh0), (fh, gh), (np.pi, 1), (Normal(), Normal()))
obs = Observable((go, fo), (1,), Normal())

ssm = StateSpaceModel(sinus, obs)

x, y = ssm.sample(500)

fig, ax = plt.subplots(2)
ax[0].plot(y)
ax[1].plot(x)

# ===== INFER VALUES ===== #

sinus = Base((fh0, gh0), (fh, gh), (Gamma(1), 1), (Normal(), Normal()))
obs = Observable((go, fo), (Gamma(1),), Normal())

ssm = StateSpaceModel(sinus, obs)

rapf = RAPF(ssm, 10000).initialize()

rapf = rapf.longfilter(y)

ax[1].plot(rapf.filtermeans())

# ===== PLOT KDE ===== #

fig2, ax2 = plt.subplots(2)

sigma = pd.Series(ssm.hidden[0].theta[0])
beta = pd.Series(ssm.observable.theta[0])

sigma.plot(kind='kde', ax=ax2[0])
beta.plot(kind='kde', ax=ax2[1])

plt.show()
