# pyfilter
`pyfilter` is a package designed for joint parameter and state inference in state space models using
particle filters and particle filter based inference algorithms. 

## Features
`pyfilter` features:
1. Particle filters in the form of [SISR](https://en.wikipedia.org/wiki/Particle_filter) and [APF](https://en.wikipedia.org/wiki/Auxiliary_particle_filter) together with different proposal distributions.
2. Both online and offline particle filter based inference algorithms such as
   1. [SMC2](https://arxiv.org/abs/1101.1528) 
   2. [NESS](https://arxiv.org/abs/1308.1883)
   3. [SMC2FW](https://arxiv.org/pdf/1503.00266.pdf)
   4. [PMMH](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf)
3. [pytorch](https://pytorch.org/) integration enables GPU accelerated inference - what took hours on a CPU now takes minutes (or even seconds).

## Requirements
`pyfilter` requires
1. [pytorch](https://pytorch.org/)
2. [pyro](https://pyro.ai)
3. [stoch-proc](https://github.com/tingiskhan/stoch-proc)

Item 3. was previously integrated in `pyfilter` but is now a standalone package.

## Example

All examples are located [here](./examples), but you'll find a short one below

```python
from stochproc import timeseries as ts, distributions as dists
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
from pyfilter.filters.particle import APF
from math import sqrt


def f(x, gamma, sigma):
    return torch.sin(x.values - gamma), sigma


def build_observation(x, a, s):
    return Normal(loc=a * x.values, scale=s)


dt = 0.1

gamma = 0.0
sigma = 1.0

init_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
inc_dist = dists.DistributionModule(Normal, loc=0.0, scale=sqrt(dt))

sine_diffusion = ts.AffineEulerMaruyama(f, (gamma, sigma), init_dist, inc_dist, dt=dt)

a = 1.0
s = 0.1

ssm = ts.StateSpaceModel(sine_diffusion, build_observation, (a, s))

sample_result = ssm.sample_states(250)
x, y = sample_result.get_paths()

fig, ax = plt.subplots()

ax.set_title("Latent")
ax.plot(sample_result.time_indexes, x, label="True", color="gray")
ax.plot(sample_result.time_indexes, y, marker="o", linestyle="None", label="Observed", color="lightblue")

filt = APF(ssm, 1_000)
result = filt.batch_filter(y)

ax.plot(sample_result.time_indexes, result.filter_means.numpy()[1:], label="Filtered", color="salmon", alpha=0.75)
ax.legend()
```

![alt text](./static/filtering.jpg?raw=true)

## Installation
`pyfilter` is currently unavailable on pypi, as such install it via
```
pip install git+https://github.com/tingiskhan/pyfilter.git
```

## Caveats
Please note that this is a project I work on in my spare time, as such there might be errors in the implementations and
sub-optimal performance. You are more than welcome to report bugs should you try out the library.

