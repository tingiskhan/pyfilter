from typing import Callable, TypeVar
from ....timeseries import AffineEulerMaruyama, AffineProcess, StochasticProcess, NewState

T = TypeVar("T", bound=StochasticProcess)
_RESULT = Callable[[T, NewState], NewState]


def _affine_process(mod: AffineProcess, state: NewState) -> NewState:
    loc, _ = mod.mean_scale(state)
    return state.propagate_from(values=loc)


def _affine_euler(mod: AffineEulerMaruyama, state: NewState) -> NewState:
    return mod.propagate_conditional(state, 0.0)


def _missing(mod, state):
    raise Exception("You didn't pass a custom function, and couldn't find a suitable pre-defined one!")


def get_pre_weight_func(func: _RESULT, process: StochasticProcess) -> _RESULT:
    """
    Gets function for generating a pre-weight for the APF.

    Args:
        func: Whether to override the choosing by passing your own custom function, else defaults to pre-defined ones.
        process: The process for which to choose a pre-weighting function for.

    Returns:
        Returns the function.
    """

    if func is not None:
        return func

    if isinstance(process, AffineEulerMaruyama):
        return _affine_euler

    if isinstance(process, AffineProcess):
        return _affine_process

    return _missing
