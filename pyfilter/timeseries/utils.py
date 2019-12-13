from .statevariable import StateVariable
import torch


def to_state_variable(obj, x):
    """
    Helper function for casting as StateVariable.
    :param obj: The timeseries object
    :type obj: StochasticProcess
    :param x: The tensor
    :type x: torch.Tensor
    :return:
    """

    if obj._inputdim > 1 and not isinstance(x, StateVariable):
        return StateVariable(x)

    return x


def tensor_caster(func):
    """
    Function for helping out when it comes to multivariate models. Returns a torch.Tensor
    :param func: The function to pass
    :type func: callable
    :rtype: torch.Tensor
    """

    def wrapper(obj, x):
        tx = to_state_variable(obj, x)

        res = func(obj, tx)
        if not isinstance(res, torch.Tensor):
            raise ValueError(f'Must be of instance {torch.Tensor.__class__.__name__}')

        res = res if not isinstance(res, StateVariable) else res.get_base()
        res.__sv = tx   # To keep GC from collecting the variable recording the gradients - really ugly, but works

        return res

    return wrapper


def finite_decorator(func):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)

        mask = torch.isfinite(out)

        if (~mask).all():
            raise ValueError('All weights seem to be `nan`, adjust your model')

        out[~mask] = float('-inf')

        return out

    return wrapper