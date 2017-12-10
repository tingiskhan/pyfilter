from pyfilter.timeseries import StateSpaceModel, Base, Observable
from pyfilter.filters import Linearized, NESS
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
obs = Observable((go, fo), (0.1,), Normal())

ssm = StateSpaceModel(sinus, obs)

predictions = 40

x, y = ssm.sample(500 + predictions)

fig, ax = plt.subplots(2)
ax[0].plot(y)
ax[1].plot(x)

# ===== INFER VALUES ===== #

sinus = Base((fh0, gh0), (fh, gh), (Gamma(1), 1), (Normal(), Normal()))
obs = Observable((go, fo), (Gamma(1),), Normal())

ssm = StateSpaceModel(sinus, obs)

alg = NESS(ssm, (300, 300), filt=Linearized).initialize()

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