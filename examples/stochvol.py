from pyfilter.timeseries import StateSpaceModel, EulerMaruyma, Observable
from pyfilter.filters import NESSMC2, Linearized
from pyfilter.distributions.continuous import Gamma, Normal
import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import time


def fh0(reversion, level, std):
    return level


def gh0(reversion, level, std):
    return std / np.sqrt(2 * reversion)


def fh(x, reversion, level, std):
    # return reversion * (level * np.exp(-x) - 1) - std ** 2 / 2
    return reversion * (level - x)


def gh(x, reversion, level, std):
    return std


def go(vol, level):
    return level


def fo(vol, level):
    return np.exp(vol)


# ===== GET DATA ===== #

fig, ax = plt.subplots(2)

stock = 'MSFT'
y = np.log(quandl.get('WIKI/{:s}'.format(stock), start_date='2010-01-01', column_index=11, transform='rdiff', api_key='zJpFs_mvKKNi1-Kse1kx') + 1)
y *= 100


# ===== DEFINE MODEL ===== #

volparams = Gamma(4, scale=0.1), Normal(0, 1), Gamma(4, scale=0.1)
logvol = EulerMaruyma((fh0, gh0), (fh, gh), volparams, (Normal(), Normal()))
obs = Observable((go, fo), (Normal(),), Normal())

ssm = StateSpaceModel(logvol, obs)

# ===== INFER VALUES ===== #

alg = NESSMC2(ssm, (400, 400)).initialize()

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

ax[1].plot(y.index[:-predictions], np.exp(np.array(alg.filtermeans())))

plt.legend()

# ===== PLOT KDEs ===== #

fig2, ax2 = plt.subplots(4)
mu = pd.DataFrame(ssm.observable.theta[0])
kappa = pd.DataFrame(ssm.hidden.theta[0])
gamma = pd.DataFrame(ssm.hidden.theta[1])
sigma = pd.DataFrame(ssm.hidden.theta[2])

mu.plot(kind='kde', ax=ax2[0])
kappa.plot(kind='kde', ax=ax2[1])
gamma.plot(kind='kde', ax=ax2[2])
sigma.plot(kind='kde', ax=ax2[3])


fig3, ax3 = plt.subplots()

pd.Series(ascum[-1]).plot(kind='kde', ax=ax3)
ax3.plot(y.iloc[-predictions:].sum(), 0, 'ro')

plt.show()
