# pyfilter
`pyfilter` is a package designed for joint parameter and state inference in (mainly) non-linear state space models using
Sequential Monte Carlo and variational inference. It is similar to `pomp`, but implemented in `Python` leveraging
[`pytorch`](https://pytorch.org/). The interface is heavily inspired by [`pymc3`](https://github.com/pymc-devs/pymc3). 

## Installation
Install the package by typing the following in a `git shell` or similar
```
pip install git+https://github.com/tingiskhan/pyfilter.git
```

## Implementations
Below is a list of implemented algorithms/filters.

### Filters
The currently implemented filters are
1. SISR
2. APF
3. UKF

For filters 1. and 2. there exist different proposals, s.a.
1. Optimal proposal when observations are linear combinations of states, and normally distributed.
2. Locally linearized observation density, mainly used for models having log-concave observation density.
4. Unscented proposal of van der Merwe et al.

### Algorithms
The currently implemented algorithms are
1. [NESS](https://arxiv.org/abs/1308.1883)
2. [SMC2](https://arxiv.org/abs/1101.1528) (see [here](https://github.com/nchopin/particles) for one of the original authors' implementation)
3. A preliminary version of Iterated Filter (IF2) by Ionides et al.
4. Variational Bayes - currently only `MeanField` is implemented
5. [SMC2FW](https://arxiv.org/pdf/1503.00266.pdf)

## Caveats
Please note that this is a project I work on in my spare time, as such there might be errors in the implementations and
sub-optimal performance. You are more than welcome to report bugs should you try out the library.

