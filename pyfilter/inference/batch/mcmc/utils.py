import torch
from ....filters import BaseFilter
from ....constants import INFTY
from ...utils import params_to_tensor, parameters_and_priors_from_model, params_from_tensor, sample_model


def seed(filter_: BaseFilter, y: torch.Tensor, num_seeds: int, num_chains) -> BaseFilter:
    seed_filter = filter_.copy()

    num_samples = num_chains * num_seeds
    seed_filter.set_nparallel(num_samples)

    sample_model(seed_filter.ssm, (num_samples, 1))

    res = seed_filter.longfilter(y, bar=False)

    params = params_to_tensor(seed_filter.ssm)
    log_likelihood = res.loglikelihood

    log_likelihood = log_likelihood.view(-1)
    log_likelihood[~torch.isfinite(log_likelihood)] = -INFTY

    best_ll = log_likelihood.argmax()

    num_params = sum(p.get_numel(constrained=True) for _, p in parameters_and_priors_from_model(filter_.ssm))
    best_params = params[best_ll]

    filter_.set_nparallel(num_chains)

    sample_model(filter_.ssm, (num_chains, 1))
    params_from_tensor(filter_.ssm, best_params.expand(num_chains, num_params), constrained=False)

    return filter_
