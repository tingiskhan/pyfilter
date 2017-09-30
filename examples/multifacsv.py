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


def fh0(reversion1, level, std1, reversion2, std2, gamma):
    return [level, level, np.zeros_like(reversion1)]


def gh0(reversion1, level, std1, reversion2, std2, gamma):
    mat = np.zeros((3, 3, *level.shape))

    mat[0, 0] = std1 / np.sqrt(2 * reversion1)
    mat[1, 1] = std2 / np.sqrt(2 * reversion2)
    mat[2, 2] = gamma

    return mat


def fh(x, reversion1, level, std1, reversion2, std2, gamma):
    out = x.copy()

    out[0] = x[0] * np.exp(-reversion1) + level * (1 - np.exp(-reversion1))
    out[1] = x[1] * np.exp(-reversion2) + level * (1 - np.exp(-reversion2))
    out[2] = x[2]

    return out


def gh(x, reversion1, level, std1, reversion2, std2, gamma):
    out = np.zeros((x.shape[0], *x.shape))

    out[0, 0] = std1 / np.sqrt(2 * reversion1) * np.sqrt(1 - np.exp(-2 * reversion1))
    out[1, 1] = std2 / np.sqrt(2 * reversion2) * np.sqrt(1 - np.exp(-2 * reversion2))
    out[2, 2] = gamma

    return out


def go(x):
    return x[2]


def fo(x):
    return np.sqrt(np.exp(x[0]) + np.exp(x[1]))


# ===== GET DATA ===== #

fig, ax = plt.subplots(4)

stock = 'msft'
y = np.log(quandl.get('WIKI/{:s}'.format(stock), start_date='2010-01-01', column_index=11, transform='rdiff') + 1)
y *= 100


# ===== DEFINE MODEL ===== #

dists = MultivariateNormal(ndim=3), MultivariateNormal(ndim=3)
params = Beta(1, 5), Normal(scale=0.3), Gamma(0.5), Beta(5, 1), Gamma(0.5), Gamma(0.25)

logvol = Base((fh0, gh0), (fh, gh), params, dists)
obs = Observable((go, fo), (), Normal())

ssm = StateSpaceModel(logvol, obs)

# ===== INFER VALUES ===== #

alg = NESS(ssm, (300, 300)).initialize()

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

ax[1].plot(y.index[:-predictions], np.exp([x[:-1] / 2 for tx in alg.filtermeans() for x in tx]))
ax[2].plot(y.index[:-predictions], [x[-1] for tx in alg.filtermeans() for x in tx])
y.iloc[:-predictions].plot(ax=ax[-1])

plt.legend()

# ===== PLOT KDEs ===== #

fig2, ax2 = plt.subplots(4)
# mu = pd.DataFrame(ssm.observable.theta[0])
kappa = pd.DataFrame(ssm.hidden[0].theta[0])
gamma = pd.DataFrame(ssm.hidden[0].theta[1])
sigma = pd.DataFrame(ssm.hidden[0].theta[2])

# mu.plot(kind='kde', ax=ax2[0])
kappa.plot(kind='kde', ax=ax2[1])
gamma.plot(kind='kde', ax=ax2[2])
sigma.plot(kind='kde', ax=ax2[3])


fig3, ax3 = plt.subplots()

pd.Series(ascum[-1]).plot(kind='kde', ax=ax3)
ax3.plot(y.iloc[-predictions:].sum(), 0, 'ro')

plt.show()
