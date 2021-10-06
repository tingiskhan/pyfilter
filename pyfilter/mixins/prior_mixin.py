import torch
from typing import Iterable, Tuple
from ..parameter import PriorBoundParameter


PRIOR_PREFIX = "prior__"
MODULE_SEPARATOR = "."


def _parameter_recursion(
    obj: "PriorMixin", parameter: PriorBoundParameter, name: str
) -> Tuple[PriorBoundParameter, "DistributionWrapper"]:
    if MODULE_SEPARATOR not in name:
        return parameter, obj._modules[f"{PRIOR_PREFIX}{name}"]

    split_name = name.split(MODULE_SEPARATOR)
    return _parameter_recursion(obj._modules[split_name[0]], parameter, MODULE_SEPARATOR.join(split_name[1:]))


class AllowPriorMixin(object):
    """
    Mixin for modules that allow specifying priors on parameters.
    """

    def register_prior(self, name, prior):
        """
        Registers a prior for a given parameter by registering ``prior`` as a module on ``self`` with naming convention
        ``{PRIOR_PREFIX}{name}, and a ``pyfilter.parameter.PriorBoundParameter`` with ``name``.
        
        Args:
            name: The name to use for the prior.
            prior: The prior of the parameter.
        """

        prior_name = f"{PRIOR_PREFIX}{name}"
        self.add_module(prior_name, prior)
        self.register_parameter(name, PriorBoundParameter(prior().sample(), requires_grad=False))

    def parameters_and_priors(self) -> Iterable[Tuple[PriorBoundParameter, "DistributionWrapper"]]:
        """
        Returns the priors and parameters of the module as an iterable of tuples, i.e::
            [(prior_parameter_0, parameter_0), ..., (prior_parameter_n, parameter_n)]
        """

        for n, p in self.named_parameters():
            yield _parameter_recursion(self, p, n)

    def priors(self):
        """
        Same as ``.parameters_and_priors()`` but only returns the priors.
        """

        for _, m in self.parameters_and_priors():
            yield m

    def sample_params(self, shape: torch.Size):
        """
        Samples the parameters of the model in place.

        Args:
            shape: The shape of the parameters to use when sampling.
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
