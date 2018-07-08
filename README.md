# Documentation

## Purpose
This package aims at providing a simple interface for users to define and perform inference on non-linear timeseries
models. It does not aim at being a replacement for any existing package. User interface is heavily inspired by [PyMC3](https://docs.pymc.io/).

This library is just a hobby of mine to keep me up to date with some of the modern filtering techniques as well as a
fun exercise in coding.

Note that I am not affiliated with or the author of any of the algorithms contained in the package. As such, there
might be errors in the implementation.

## Installing the package
Install the package by typing the following in a `git shell` or similar
```
pip install git+https://github.com/tingiskhan/pyfilter.git
```

## Implemented algorithms
The current implementation supports:
1. Particle filter
    1. Bootstrap proposal
    2. Linearized proposal
    3. Unscented proposal
2. Auxiliary Particle filter
3. Unscented Kalman filter
4. Kalman-Laplace filter
5. Nested particle filters for online parameter estimation in discrete-time state-space Markov models (NESS)
6. SMC2: an efficient algorithm for sequential analysis of state space models (SMC2)
7. Liu-West filter

## Future implementations
Some future functionality to be implemented might be:
1. Quasi Monte Carlo 
2. Dual/Joint Unscented Kalman Filter (or Kalman-Laplace)

## Using the package
For examples on how to define models and performing inference please see
[`examples`](https://github.com/tingiskhan/pyfilter/tree/master/examples), it should be rather straightforward from there.

