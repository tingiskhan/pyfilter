from torch.distributions import MultivariateNormal
import warnings
from typing import Callable
from torch.distributions import Distribution, Independent
import torch
from typing import Tuple, Union
from ..filters import BaseFilter, FilterResult
from .state import AlgorithmState
from ..timeseries import StateSpaceModel, StochasticProcess
from ..distributions import Prior
from ..parameter import ExtendedParameter
from ..constants import EPS


PropConstructor = Callable[[AlgorithmState, BaseFilter, torch.Tensor], Distribution]
Process = Union[StochasticProcess, StateSpaceModel]


def _construct_mvn(x: torch.Tensor, w: torch.Tensor, scale=1.0):
    """
    Constructs a multivariate normal distribution from weighted samples.
    """

    mean = (x * w.unsqueeze(-1)).sum(0)
    centralized = x - mean
    cov = torch.matmul(w * centralized.t(), centralized)

    if cov.det() == 0.0:
        chol = cov.diag().sqrt().diag()
    else:
        chol = cov.cholesky()

    return MultivariateNormal(mean, scale_tril=scale * chol)


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


def run_pmmh(
    filter_: BaseFilter,
    state: FilterResult,
    prop_kernel: Distribution,
    prop_filt,
    y: torch.Tensor,
    size=torch.Size([]),
    **kwargs,
) -> Tuple[torch.Tensor, FilterResult, BaseFilter]:
    """
    Runs one iteration of a vectorized Particle Marginal Metropolis hastings.
    """

    rvs = prop_kernel.sample(size)
    params_from_tensor(prop_filt.ssm, rvs, constrained=False)

    new_res = prop_filt.longfilter(y, bar=False, **kwargs)

    diff_logl = new_res.loglikelihood - state.loglikelihood
    diff_prior = (eval_prior_log_prob(prop_filt.ssm, False) - eval_prior_log_prob(filter_.ssm, False)).squeeze()

    if isinstance(prop_kernel, Independent) and size == torch.Size([]):
        diff_prop = 0.0
    else:
        diff_prop = prop_kernel.log_prob(params_to_tensor(filter_.ssm, constrained=False)) - prop_kernel.log_prob(rvs)

    log_acc_prob = diff_prop + diff_prior + diff_logl
    res: torch.Tensor = torch.empty_like(log_acc_prob).uniform_().log() < log_acc_prob

    return res, new_res, prop_filt
