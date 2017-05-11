from pyfilter.model import StateSpaceModel
from pyfilter.timeseries.meta import Base
from pyfilter.filters.rapf import RAPF
from pyfilter.distributions.continuous import Gamma, Normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data


def fh0(kappa, gamma, sigma):
    return gamma


def gh0(kappa, gamma, sigma):
    return sigma / np.sqrt(2 * kappa)


def fh(x, kappa, gamma, sigma):
    return x + kappa * (gamma - x)


def gh(x, kappa, gamma, sigma):
    return sigma


def go(vol, mu):
    return mu


def fo(vol, mu):
    return np.exp(vol / 2)

# ===== INFER VALUES ===== #

fig, ax = plt.subplots(2)

stock = data.DataReader('^gspc', 'yahoo', start='2010-01-01')

y = stock['Adj Close'].pct_change().iloc[1:] * 100

ax[0].plot(y)

logvol = Base((fh0, gh0), (fh, gh), (Gamma(1), Normal(), Gamma(1)), (Normal(), Normal()))

obs = Base((None, None), (go, fo), (Normal(),), (None, Normal()))

ssm = StateSpaceModel(logvol, obs)

rapf = RAPF(ssm, 30000).initialize()

rapf = rapf.longfilter(y)

ax[1].plot(rapf.filtermeans())


# ===== PLOT KDE ===== #

fig2, ax2 = plt.subplots(4)

mu = pd.Series(ssm.observable.theta[0])
kappa = pd.Series(ssm.hidden[0].theta[0])
gamma = pd.Series(ssm.hidden[0].theta[1])
sigma = pd.Series(ssm.hidden[0].theta[2])

mu.plot(kind='kde', ax=ax2[0])
kappa.plot(kind='kde', ax=ax2[1])
gamma.plot(kind='kde', ax=ax2[2])
sigma.plot(kind='kde', ax=ax2[3])

plt.show()
