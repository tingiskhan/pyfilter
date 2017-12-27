from pyfilter.timeseries import StateSpaceModel, EulerMaruyma, Observable
from pyfilter.filters import NESS, UKF
from pyfilter.distributions.continuous import Gamma, Normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def fh0(alpha, sigma):
    return 0


def gh0(alpha, sigma):
    return sigma


def fh(x, alpha, sigma):
    return np.sin(x - alpha)


def gh(x, alpha, sigma):
    return sigma


def go(x, beta):
    return x


def fo(x, beta):
    return beta

# ===== SIMULATE SSM ===== #

sinus = EulerMaruyma((fh0, gh0), (fh, gh), (np.pi, 1), (Normal(), Normal()))
obs = Observable((go, fo), (0.6,), Normal())

ssm = StateSpaceModel(sinus, obs)

predictions = 40

x, y = ssm.sample(500 + predictions)

fig, ax = plt.subplots(2)
ax[0].plot(y)
ax[1].plot(x)

# ===== INFER VALUES ===== #

sinus = EulerMaruyma((fh0, gh0), (fh, gh), (Gamma(1), 1), (Normal(), Normal()))
obs = Observable((go, fo), (Gamma(1),), Normal())

ssm = StateSpaceModel(sinus, obs)

alg = NESS(ssm, (1000,), filt=UKF).initialize()

alg = alg.longfilter(y[:-predictions])

ax[1].plot(alg.filtermeans())

# ===== PLOT KDE ===== #

fig2, ax2 = plt.subplots(3)

sigma = pd.DataFrame(ssm.hidden.theta[0])
beta = pd.DataFrame(ssm.observable.theta[0])

sigma.plot(kind='kde', ax=ax2[0])
beta.plot(kind='kde', ax=ax2[1])

ax2[2].plot(alg._filter.s_n)

plt.show()