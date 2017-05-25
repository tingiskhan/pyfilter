from pyfilter.model import StateSpaceModel
from pyfilter.timeseries.meta import Base
from pyfilter.timeseries.observable import Observable
from pyfilter.filters.rapf import RAPF
from pyfilter.distributions.continuous import Gamma, Normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data
import quandl


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

y = quandl.get('WIKI/AAPL', start_date='2010-01-01', api_key='<APIKEY>', column_index=11, transform='rdiff')
y *= 100

ax[0].plot(y)

predictions = 30

logvol = Base((fh0, gh0), (fh, gh), (Gamma(1), Normal(), Gamma(1)), (Normal(), Normal()))

obs = Observable((go, fo), (Normal(),), Normal())

ssm = StateSpaceModel(logvol, obs)

rapf = RAPF(ssm, 3000).initialize()

rapf = rapf.longfilter(y[:-predictions])

ax[1].plot(rapf.filtermeans())

p_x, p_y = rapf.predict(predictions)

ax[0].plot(y.index[-predictions:], p_y, alpha=0.03, color='r')

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
