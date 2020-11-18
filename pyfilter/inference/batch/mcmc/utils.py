from ....filters import BaseFilter
import torch
from ....constants import INFTY


def seed(filter_: BaseFilter, y: torch.Tensor, num_seeds: int, num_chains) -> BaseFilter:
    # ===== Construct and run filter ===== #
    seed_filter = filter_.copy()

    num_samples = num_chains * num_seeds
    seed_filter.set_nparallel(num_samples)
    seed_filter.ssm.sample_params(num_samples)
    seed_filter.ssm.viewify_params((num_samples, 1))

    res = seed_filter.longfilter(y, bar=False)

    # ===== Find best parameters ===== #
    params = seed_filter.ssm.parameters_to_array()
    log_likelihood = res.loglikelihood

    log_likelihood = log_likelihood.view(-1)
    log_likelihood[~torch.isfinite(log_likelihood)] = -INFTY

    best_ll = log_likelihood.argmax()

    num_params = sum(p.numel_() for p in filter_.ssm.trainable_parameters)
    best_params = params[best_ll]

    # ===== Set filter parameters ===== #
    filter_.set_nparallel(num_chains)
    filter_.ssm.sample_params(num_chains)
    filter_.ssm.viewify_params((num_chains, 1))
    filter_.ssm.parameters_from_array(best_params.expand(num_chains, num_params))

    return filter_
