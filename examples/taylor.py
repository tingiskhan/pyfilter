from pyfilter.model import StateSpaceModel
from pyfilter.timeseries.meta import Base
from pyfilter.timeseries.observable import Observable
from pyfilter.filters import NESSMC2
from pyfilter.distributions.continuous import Gamma, Normal, Beta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def fh0(alpha, sigma):
    return 0


def gh0(alpha, sigma):
    return sigma


def fh(x, alpha, sigma):
    return alpha * x


def gh(x, alpha, sigma):
    return sigma


def go(x, beta):
    return 0


def fo(x, beta):
    return beta * np.exp(x / 2)

# ===== SIMULATE SSM ===== #
np.random.seed(123)
logvol = Base((fh0, gh0), (fh, gh), (0.99, 0.25), (Normal(), Normal()))
obs = Observable((go, fo), (0.6,), Normal())

ssm = StateSpaceModel(logvol, obs)

x, y = ssm.sample(1000)

fig, ax = plt.subplots(2)
ax[0].plot(y)
ax[1].plot(x)

# ===== INFER VALUES ===== #

logvol = Base((fh0, gh0), (fh, gh), (Beta(5, 1), Gamma(1)), (Normal(), Normal()))
obs = Observable((go, fo), (Gamma(0.5),), Normal())

ssm = StateSpaceModel(logvol, obs)

rapf = NESSMC2(ssm, (300, 300), handshake=0.4).initialize()

rapf = rapf.longfilter(y)

# ax[1].plot(rapf.filtermeans())

# ===== PLOT KDE ===== #

fig2, ax2 = plt.subplots(3)

alpha = pd.DataFrame(ssm.hidden[0].theta[0])
sigma = pd.DataFrame(ssm.hidden[0].theta[1])
beta = pd.DataFrame(ssm.observable.theta[0])

alpha.plot(kind='kde', ax=ax2[0])
sigma.plot(kind='kde', ax=ax2[1])
beta.plot(kind='kde', ax=ax2[2])

plt.show()
