import torch
from typing import Iterable, Tuple
from ..parameter import ExtendedParameter


PRIOR_PREFIX = "prior__"
MODULE_SEPARATOR = "."


def _parameter_recursion(
    obj: "PriorMixin", parameter: ExtendedParameter, name: str
) -> Tuple[ExtendedParameter, "DistributionWrapper"]:
    if MODULE_SEPARATOR not in name:
        return parameter, obj._modules[f"{PRIOR_PREFIX}{name}"]

    split_name = name.split(MODULE_SEPARATOR)
    return _parameter_recursion(obj._modules[split_name[0]], parameter, MODULE_SEPARATOR.join(split_name[1:]))


class PriorMixin(object):
    def register_prior(self, name, prior):
        prior_name = f"{PRIOR_PREFIX}{name}"
        self.add_module(prior_name, prior)
        self.register_parameter(name, ExtendedParameter(prior().sample(), requires_grad=False))

    def parameters_and_priors(self) -> Iterable[Tuple[ExtendedParameter, "DistributionWrapper"]]:
        for n, p in self.named_parameters():
            yield _parameter_recursion(self, p, n)

    def priors(self):
        for _, m in self.parameters_and_priors():
            yield m

    # TODO: Fix static type checking
    def sample_params(self, shape):
        """
        Samples the parameters of the model in place.
        """

        for param, prior in self.parameters_and_priors():
            param.sample_(prior, shape)

        return self

    def eval_prior_log_prob(self, constrained=True) -> torch.Tensor:
        """
        Calculates the prior log-likelihood of the current values of the parameters.

        :param constrained: If you use an unconstrained proposal you need to use `transformed=True`
        """

        return sum((prior.eval_prior(p, constrained) for p, prior in self.parameters_and_priors()))
