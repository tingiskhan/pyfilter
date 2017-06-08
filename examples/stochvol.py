from pyfilter.model import StateSpaceModel
from pyfilter.timeseries.meta import Base
from pyfilter.timeseries.observable import Observable
from pyfilter.filters.ness import NESS
from pyfilter.distributions.continuous import Gamma, Normal
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
    return x + reversion * (level - x)


def gh(x, reversion, level, std):
    return std


def go(vol, level):
    return level


def fo(vol, level):
    return np.exp(vol / 2)


# ===== GET DATA ===== #

fig, ax = plt.subplots()

stock = 'AAPL'
y = quandl.get('WIKI/{:s}'.format(stock), start_date='2012-01-01', column_index=11, transform='rdiff')
y *= 100


# ===== DEFINE MODEL ===== #

logvol = Base((fh0, gh0), (fh, gh), (Gamma(1), Normal(), Gamma(1)), (Normal(), Normal()))
mu = Base((fh0, gh0), (fh, gh), (Gamma(1), Normal(), Gamma(1)), (Normal(), Normal()))
obs = Observable((go, fo), (), Normal())

ssm = StateSpaceModel((logvol, mu), obs)

# ===== INFER VALUES ===== #

alg = NESS(ssm, (500, 500)).initialize()

predictions = 30

start = time.time()
alg = alg.longfilter(y[:-predictions])
print('Took {:.1f} seconds to finish'.format(time.time() - start))

# ===== PREDICT ===== #

p_x, p_y = alg.predict(predictions)

ascum = np.cumsum(np.array(p_y), axis=0)

up = np.percentile(ascum, 95, axis=1)
down = np.percentile(ascum, 5, axis=1)

ax.plot(y.index[-predictions:], up, alpha=0.6, color='r', label='95%')
ax.plot(y.index[-predictions:], down, alpha=0.6, color='r', label='5%')
ax.plot(y.index[-predictions:], ascum.mean(axis=1), color='b', label='Mean')

actual = y.iloc[-predictions:].cumsum()
ax.plot(y.index[-predictions:], actual, color='g', label='Actual')

plt.legend()

# ===== PLOT KDEs ===== #

fig2, ax2 = plt.subplots(3, 2)

kappa = pd.DataFrame(ssm.hidden[0].theta[0])
gamma = pd.DataFrame(ssm.hidden[0].theta[1])
sigma = pd.DataFrame(ssm.hidden[0].theta[2])

alpha = pd.DataFrame(ssm.hidden[1].theta[0])
beta = pd.DataFrame(ssm.hidden[1].theta[1])
delta = pd.DataFrame(ssm.hidden[1].theta[2])

kappa.plot(kind='kde', ax=ax2[0, 0])
gamma.plot(kind='kde', ax=ax2[1, 0])
sigma.plot(kind='kde', ax=ax2[2, 0])

alpha.plot(kind='kde', ax=ax2[0, 1])
beta.plot(kind='kde', ax=ax2[1, 1])
delta.plot(kind='kde', ax=ax2[2, 1])

plt.show()
