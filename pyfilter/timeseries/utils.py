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

    def wrapper(obj, x, **kwargs):
        tx = to_state_variable(obj, x)

        res = func(obj, tx, **kwargs)

        if not isinstance(res, StateVariable):
            return res

        res = res.get_base()
        res.__sv = tx   # To keep GC from collecting the variable recording the gradients - really ugly, but works

        return res

    return wrapper


def tensor_caster_mult(func):
    """
    Function for helping out calculating the
    :param func: The function to pass
    :type func: callable
    :rtype: torch.Tensor
    """

    def wrapper(obj, y, x):
        tx = to_state_variable(obj, x)

        res = func(obj, y, tx)
        if not isinstance(res, torch.Tensor):
            raise ValueError(f'Must be of instance {torch.Tensor.__class__.__name__}')

        res.__sv = tx  # To keep GC from collecting the variable recording the gradients - really ugly, but works

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