from pyfilter.model import StateSpaceModel
from pyfilter.timeseries.meta import Base
from pyfilter.filters.rapf import RAPF
from pyfilter.distributions.continuous import Gamma, Normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

# ===== SIMULATE SSM ===== #

# returns = Base((fh0, gh0), (fh, gh), (0.8, 0, 0.1), (Normal(), Normal()))
logvol = Base((fh0, gh0), (fh, gh), (0.1, 0, 0.3), (Normal(), Normal()))
obs = Base((None, None), (go, fo), (0.05,), (None, Normal()))

ssm = StateSpaceModel(logvol, obs)

x, y = ssm.sample(500)

fig, ax = plt.subplots(2)
ax[0].plot(y)
ax[1].plot(x)

# ===== INFER VALUES ===== #

logvol = Base((fh0, gh0), (fh, gh), (Gamma(1), Normal(), Gamma(1)), (Normal(), Normal()))
# returns = Base((fh0, gh0), (fh, gh), (Gamma(1), Normal(), Gamma(1)), (Normal(), Normal()))

obs = Base((None, None), (go, fo), (Normal(),), (None, Normal()))

ssm = StateSpaceModel(logvol, obs)

rapf = RAPF(ssm, 30000).initialize()

rapf = rapf.longfilter(y)

ax[1].plot(rapf.filtermeans())


# ===== PLOT KDE ===== #

fig2, ax2 = plt.subplots(3)

kappa = pd.Series(ssm.hidden[0].theta[0])
gamma = pd.Series(ssm.hidden[0].theta[1])
sigma = pd.Series(ssm.hidden[0].theta[2])


kappa.plot(kind='kde', ax=ax2[0])
gamma.plot(kind='kde', ax=ax2[1])
kappa.plot(kind='kde', ax=ax2[2])

# fig3, ax3 = plt.subplots(3)
#
# for i, p in enumerate(ssm.hidden[1].theta):
#     param = pd.Series(p)
#     param.plot(kind='kde', ax=ax3[i])

plt.show()
