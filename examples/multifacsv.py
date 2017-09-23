import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl
from pyfilter.distributions.continuous import Gamma, Normal, Beta, MultivariateNormal
from pyfilter.filters import NESS
from pyfilter.timeseries import Base
from pyfilter.timeseries import Observable
from pyfilter.timeseries import StateSpaceModel


def fh0(reversion1, level, std1, reversion2, std2):
    return [level, level]


def gh0(reversion1, level, std1, reversion2, std2):
    mat = np.zeros((2, 2, *level.shape))

    mat[0, 0] = std1 / np.sqrt(2 * reversion1)
    mat[1, 1] = std2 / np.sqrt(2 * reversion2)

    return mat


def fh(x, reversion1, level, std1, reversion2, std2):
    out = x.copy()

    out[0] = x[0] * np.exp(-reversion1) + level * (1 - np.exp(-reversion1))
    out[1] = x[1] * np.exp(-reversion2) + level * (1 - np.exp(-reversion2))

    return out


def gh(x, reversion1, level, std1, reversion2, std2):
    out = np.zeros((x.shape[0], *x.shape))

    out[0, 0] = std1 / np.sqrt(2 * reversion1) * np.sqrt(1 - np.exp(-2 * reversion1))
    out[1, 1] = std2 / np.sqrt(2 * reversion2) * np.sqrt(1 - np.exp(-2 * reversion2))

    return out


def go(vol, level):
    return level


def fo(vol, level):
    return np.sqrt(np.exp(vol[0]) + np.exp(vol[1]))


# ===== GET DATA ===== #

fig, ax = plt.subplots(2)

stock = 'ABBV'
y = np.log(quandl.get('WIKI/{:s}'.format(stock), start_date='2010-01-01', column_index=11, transform='rdiff') + 1)
y *= 100


# ===== DEFINE MODEL ===== #

dists = (MultivariateNormal(), MultivariateNormal())
logvol = Base((fh0, gh0), (fh, gh), (Beta(1, 5), Normal(scale=0.3), Gamma(0.5), Beta(5, 1), Gamma(0.5)), dists)
obs = Observable((go, fo), (Normal(),), Normal())

ssm = StateSpaceModel(logvol, obs)

# ===== INFER VALUES ===== #

alg = NESS(ssm, (600, 600)).initialize()

predictions = 30

start = time.time()
alg = alg.longfilter(y[:-predictions])
print('Took {:.1f} seconds to finish for {:s}'.format(time.time() - start, stock))

# ===== PREDICT ===== #

p_x, p_y = alg.predict(predictions)

ascum = np.cumsum(np.array(p_y), axis=0)

up = np.percentile(ascum, 99, axis=1)
down = np.percentile(ascum, 1, axis=1)

ax[0].plot(y.index[-predictions:], up, alpha=0.6, color='r', label='95%')
ax[0].plot(y.index[-predictions:], down, alpha=0.6, color='r', label='5%')
ax[0].plot(y.index[-predictions:], ascum.mean(axis=1), color='b', label='Mean')

actual = y.iloc[-predictions:].cumsum()
ax[0].plot(y.index[-predictions:], actual, color='g', label='Actual')

ax[1].plot(y.index[:-predictions], np.exp(alg.filtermeans()))

plt.legend()

# ===== PLOT KDEs ===== #

fig2, ax2 = plt.subplots(4)
mu = pd.DataFrame(ssm.observable.theta[0])
kappa = pd.DataFrame(ssm.hidden[0].theta[0])
gamma = pd.DataFrame(ssm.hidden[0].theta[1])
sigma = pd.DataFrame(ssm.hidden[0].theta[2])

mu.plot(kind='kde', ax=ax2[0])
kappa.plot(kind='kde', ax=ax2[1])
gamma.plot(kind='kde', ax=ax2[2])
sigma.plot(kind='kde', ax=ax2[3])


fig3, ax3 = plt.subplots()

pd.Series(ascum[-1]).plot(kind='kde', ax=ax3)
ax3.plot(y.iloc[-predictions:].sum(), 0, 'ro')

plt.show()
