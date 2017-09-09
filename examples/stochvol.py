from pyfilter.model import StateSpaceModel
from pyfilter.timeseries import Base
from pyfilter.timeseries import Observable
from pyfilter.filters import NESSMC2
from pyfilter.distributions.continuous import Gamma, Normal, Beta
from pyfilter.proposals import Linearized as Linz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import time


def fh0(reversion, level, std):
    return level


def gh0(reversion, level, std):
    return std / np.sqrt(2 * reversion)


def fh(x, reversion, level, std):
    return x * np.exp(-reversion) + level * (1 - np.exp(-reversion))


def gh(x, reversion, level, std):
    return std / np.sqrt(2 * reversion) * np.sqrt(1 - np.exp(-2 * reversion))


def go(vol, level, beta):
    return level + (beta - 1) * np.exp(vol)


def fo(vol, level, beta):
    return np.exp(vol / 2)


# ===== GET DATA ===== #

fig, ax = plt.subplots()

stock = 'yhoo'
y = np.log(quandl.get('WIKI/{:s}'.format(stock), start_date='2010-01-01', column_index=11, transform='rdiff') + 1)
y *= 100


# ===== DEFINE MODEL ===== #

logvol = Base((fh0, gh0), (fh, gh), (Beta(1, 3), Normal(scale=0.3), Gamma(1)), (Normal(), Normal()))
obs = Observable((go, fo), (Normal(), 1), Normal())

ssm = StateSpaceModel(logvol, obs)

# ===== INFER VALUES ===== #

alg = NESSMC2(ssm, (400, 400), proposal=Linz).initialize()

predictions = 30

start = time.time()
alg = alg.longfilter(y[:-predictions])
print('Took {:.1f} seconds to finish for {:s}'.format(time.time() - start, stock))

# ===== PREDICT ===== #

p_x, p_y = alg.predict(predictions)

ascum = np.cumsum(np.array(p_y), axis=0)

up = np.percentile(ascum, 99, axis=1)
down = np.percentile(ascum, 1, axis=1)

ax.plot(y.index[-predictions:], up, alpha=0.6, color='r', label='95%')
ax.plot(y.index[-predictions:], down, alpha=0.6, color='r', label='5%')
ax.plot(y.index[-predictions:], ascum.mean(axis=1), color='b', label='Mean')

actual = y.iloc[-predictions:].cumsum()
ax.plot(y.index[-predictions:], actual, color='g', label='Actual')

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
