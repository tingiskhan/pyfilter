# pyfilter
`pyfilter` is a package designed for joint parameter and state inference in (mainly) non-linear state space models using
Sequential Monte Carlo and variational inference. It is similar to `pomp`, but implemented in `Python` leveraging
[`pytorch`](https://pytorch.org/). The interface is heavily inspired by [`pymc3`](https://github.com/pymc-devs/pymc3). 

## Example
```python
from pyfilter.timeseries import AffineEulerMaruyama, LinearGaussianObservations
import torch
from pyfilter.distributions import DistributionWrapper
from torch.distributions import Normal
import matplotlib.pyplot as plt
from pyfilter.filters.particle import APF, proposals as p
from math import sqrt


def drift(x, gamma, sigma):
    return torch.sin(x.state - gamma)


def diffusion(x, gamma, sigma):
    return sigma


dt = 0.1
parameters = 0.0, 1.0
inc_dist = DistributionWrapper(Normal, loc=0.0, scale=1.0)
init_dist = DistributionWrapper(Normal, loc=0.0, scale=sqrt(dt))

sinus_diffusion = AffineEulerMaruyama(
    (drift, diffusion),
    parameters,
    init_dist,
    inc_dist,
    dt=dt,
    num_steps=int(1 / dt)
)
ssm = LinearGaussianObservations(sinus_diffusion, scale=0.1)

x, y = ssm.sample_path(1000)

fig, ax = plt.subplots(2)
ax[0].plot(x.numpy(), label="True")
ax[1].plot(y.numpy())

filt = APF(ssm, 1000, proposal=p.Bootstrap())
filter_result = filt.longfilter(y)

ax[0].plot(filter_result.filter_means.numpy(), label="Filtered")
ax[0].legend()

plt.show()
```

## Installation
Install the package by typing the following in a `git shell` or similar
```
pip install git+https://github.com/tingiskhan/pyfilter.git
```

## Implementations
Below is a list of implemented algorithms/filters.

### Filters
The currently implemented filters are
1. [SISR](https://en.wikipedia.org/wiki/Particle_filter)
2. [APF](https://en.wikipedia.org/wiki/Auxiliary_particle_filter)
3. [UKF](https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf)

For filters 1. and 2. there exist different proposals, s.a.
1. Optimal proposal when observations are linear combinations of states, and normally distributed.
2. Locally linearized observation density, mainly used for models having log-concave observation density.

### Algorithms
The currently implemented algorithms are
1. [NESS](https://arxiv.org/abs/1308.1883)
2. [SMC2](https://arxiv.org/abs/1101.1528) (see [here](https://github.com/nchopin/particles) for one of the original authors' implementation)
3. [Variational Bayes](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)
4. [SMC2FW](https://arxiv.org/pdf/1503.00266.pdf)
5. [PMMH](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf)

## Caveats
Please note that this is a project I work on in my spare time, as such there might be errors in the implementations and
sub-optimal performance. You are more than welcome to report bugs should you try out the library.

