import torch
from typing import Tuple, Union
from ..timeseries import StateSpaceModel, StochasticProcess
from ..distributions import Prior
from ..parameter import PriorBoundParameter


Process = Union[StochasticProcess, StateSpaceModel]


def parameters_and_priors_from_model(model: Process) -> Tuple[Tuple[PriorBoundParameter, Prior], ...]:
    """
    Gets the parameters and priors for a given model.

    Args:
        model: The model, can be either a ``StochasticProcess`` or ``StateSpaceModel``, to get the parameters and priors
            for.

    Returns:
        Returns a tuple consisting of: ``[(parameter_0, prior_0), ...]``
    """

    if isinstance(model, StateSpaceModel):
        return tuple(model.hidden.parameters_and_priors()) + tuple(model.observable.parameters_and_priors())

    return tuple(model.parameters_and_priors())


def priors_from_model(model: StateSpaceModel) -> Tuple[Prior, ...]:
    """
    Like ``parameters_and_priors_from_model``, but only gets the priors for a given model.

    Args:
        model: See ``parameters_and_priors_from_model``.

    Returns:
        Returns a tuple consisting of: ``[prior_0, ...]``.
    """

    return tuple(prior for (p, prior) in parameters_and_priors_from_model(model))


def params_to_tensor(model: StateSpaceModel, constrained=False) -> torch.Tensor:
    """
    Given a model consisting of `n` parameters, combine the parameters into a single tensor.

    Args:
        model: See ``parameters_and_priors_from_model``.
        constrained: Whether to return constrained or unconstrained parameters.

    Returns:
        Returns a tensor combining all of the parameters, resulting size will be
        ``([batch size], combined size of all parameters)``.
    """

    parameters_and_priors = parameters_and_priors_from_model(model)

    res = tuple(
        (p if constrained else prior.get_unconstrained(p)).view(-1, prior.get_numel(constrained))
        for p, prior in parameters_and_priors
    )

    return torch.cat(res, dim=-1)


def params_from_tensor(model: Process, x: torch.Tensor, constrained=False):
    """
    Performs the inverse of ``params_to_tensor``, i.e. given a tensor of size
    ``([batch size], combined size of all parameters)``, map the values back to the parameters of ``model``.

    Args:
        model: See ``parameters_and_priors_from_model``.
        x: The parameters in tensor form.
        constrained: Whether the ``x`` consists of constrained or unconstrained parameter values.
    """

    left = 0
    for p, prior in parameters_and_priors_from_model(model):
        slc, numel = prior.get_slice_for_parameter(left, constrained)
        p.update_values(x[..., slc], prior, constrained)

        left += numel

    return


def sample_model(model: Process, shape: torch.Size):
    """
    Samples from the priors and populates the parameters with said samples.

    Args:
        model: See ``parameters_and_priors_from_model``.
        shape: The batch shape to sample
    """

    if isinstance(model, StateSpaceModel):
        model.hidden.sample_params(shape)
        model.observable.sample_params(shape)
    else:
        model.sample_params(shape)


def eval_prior_log_prob(model: Process, constrained=False) -> torch.Tensor:
    """
    Evaluates the priors' log likelihood of the current parameter values.

    Args:
        model: See ``parameters_and_priors_from_model``.
        constrained: Whether to consider the log likelihood of the parameters in constrained or unconstrained space.
    """

    if isinstance(model, StateSpaceModel):
        return model.hidden.eval_prior_log_prob(constrained) + model.observable.eval_prior_log_prob(constrained)

    return model.eval_prior_log_prob(constrained)
