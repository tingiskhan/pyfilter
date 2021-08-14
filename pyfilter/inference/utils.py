import warnings
import torch
from typing import Tuple, Union
from ..timeseries import StateSpaceModel, StochasticProcess
from ..distributions import Prior
from ..parameter import ExtendedParameter


Process = Union[StochasticProcess, StateSpaceModel]


def experimental(func):
    def wrapper(obj, *args, **kwargs):
        warnings.warn(f"{obj:s} is an experimental algorithm, use at own risk")

        return func(obj, *args, **kwargs)

    return wrapper


def preliminary(func):
    def wrapper(obj, *args, **kwargs):
        warnings.warn(f"{obj:s} is only a preliminary version algorithm, use at own risk")

        return func(obj, *args, **kwargs)

    return wrapper


def parameters_and_priors_from_model(model: Process) -> Tuple[Tuple[ExtendedParameter, Prior], ...]:
    if isinstance(model, StateSpaceModel):
        return tuple(model.hidden.parameters_and_priors()) + tuple(model.observable.parameters_and_priors())

    return tuple(model.parameters_and_priors())


def priors_from_model(model: StateSpaceModel) -> Tuple[Prior, ...]:
    return tuple(prior for (p, prior) in parameters_and_priors_from_model(model))


def params_to_tensor(model: StateSpaceModel, constrained=False) -> torch.Tensor:
    parameters_and_priors = parameters_and_priors_from_model(model)

    res = tuple(
        (p if constrained else prior.get_unconstrained(p)).view(-1, prior.get_numel(constrained))
        for p, prior in parameters_and_priors
    )

    return torch.cat(res, dim=-1)


def params_from_tensor(model: Process, x: torch.Tensor, constrained=False):
    left = 0
    for p, prior in parameters_and_priors_from_model(model):
        slc, numel = prior.get_slice_for_parameter(left, constrained)
        p.update_values(x[..., slc], prior, constrained)

        left += numel

    return


def sample_model(model: Process, shape):
    if isinstance(model, StateSpaceModel):
        model.hidden.sample_params(shape)
        model.observable.sample_params(shape)
    else:
        model.sample_params(shape)


def eval_prior_log_prob(model: Process, constrained=False):
    if isinstance(model, StateSpaceModel):
        return model.hidden.eval_prior_log_prob(constrained) + model.observable.eval_prior_log_prob(constrained)

    return model.eval_prior_log_prob(constrained)
