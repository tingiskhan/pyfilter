from pyfilter.timeseries import StateSpaceModel, EulerMaruyma, Observable
from pyfilter.filters import NESSMC2, RAPF
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
    return reversion * (level - x)


def gh(x, reversion, level, std):
    return std


def go(vol, level, sigma):
    return level


def fo(vol, level, sigma):
    return np.exp(vol / 2) + sigma


# ===== GET DATA ===== #

fig, ax = plt.subplots(2)

stock = 'MSFT'
y = np.log(quandl.get('WIKI/{:s}'.format(stock), start_date='2010-01-01', column_index=11, transform='rdiff', api_key='zJpFs_mvKKNi1-Kse1kx') + 1)
y *= 100


# ===== DEFINE MODEL ===== #

volparams = Gamma(4, scale=0.1), Normal(0, 1), Gamma(4, scale=0.1)
logvol = EulerMaruyma((fh0, gh0), (fh, gh), volparams, (Normal(), Normal()))
obs = Observable((go, fo), (Normal(), Gamma(4, scale=0.1)), Normal())

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

ax[1].plot(y.index[:-predictions], np.exp(np.array(alg.filtermeans()) / 2))

plt.legend()

# ===== PLOT KDEs ===== #

figo, axo = plt.subplots(2)
for i, p in enumerate(ssm.observable.theta):
    pd.DataFrame(p).plot(kind='kde', ax=axo[i])

figh, axh = plt.subplots(3)
for i, p in enumerate(ssm.hidden.theta):
    pd.DataFrame(p).plot(kind='kde', ax=axh[i])

figp, axp = plt.subplots()

pd.Series(ascum[-1]).plot(kind='kde', ax=axp)
axp.plot(y.iloc[-predictions:].sum(), 0, 'ro')

plt.show()
