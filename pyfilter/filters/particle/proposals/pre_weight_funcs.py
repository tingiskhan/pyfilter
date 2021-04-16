from typing import Callable
from ....timeseries import AffineEulerMaruyama, AffineProcess, StochasticProcess, NewState

_RESULT = Callable[[StochasticProcess, NewState], NewState]


def _affine_process(mod: AffineProcess, state: NewState) -> NewState:
    loc, _ = mod.mean_scale(state)
    return state.propagate_from(values=loc)


def _affine_euler(mod: AffineEulerMaruyama, state: NewState) -> NewState:
    return mod.propagate_conditional(state, 0.0)


def get_pre_weight_func(func, process: StochasticProcess) -> _RESULT:
    if func is not None:
        return func

    if isinstance(process, AffineEulerMaruyama):
        return _affine_euler

    if isinstance(process, AffineProcess):
        return _affine_process

    return None
