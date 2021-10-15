import warnings

from torch.nn import Module, ModuleDict, ParameterDict
from abc import ABC
import torch
from typing import Iterable, Tuple, Dict, Union
from pyfilter.parameter import PriorBoundParameter
from pyfilter.container import BufferDict


class HasPriorsModule(Module, ABC):
    """
    Abstract base class that allows registering priors.
    """

    def __init__(self):
        """
        Initializes the ``HasPriorsModule`` class.
        """

        super().__init__()

        self.prior_dict = ModuleDict()

        # Bug for ``torch.nn.ParameterDict`` as ``__setitem__`` is disallowed, but ``Module`` initializes training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.parameter_dict = ParameterDict()
            self.buffer_dict = BufferDict()

    def _register_parameter_or_prior(self, name: str, p):
        """
        Helper method for registering either a:
            - ``pyfilter.distributions.Prior``
            - ``torch.nn.Parameter``
            - ``torch.Tensor``

        Args:
            name: The name to use for the object.
            p: The object to register.
        """

        from .distributions import Prior

        if isinstance(p, Prior):
            self.register_prior(name, p)
        elif isinstance(p, torch.nn.Parameter):
            self.parameter_dict[name] = p
        else:
            self.buffer_dict[name] = (p if (isinstance(p, torch.Tensor) or p is None) else torch.tensor(p))

    def parameters_and_buffers(self) -> Dict[str, Union[torch.Tensor, torch.nn.Parameter]]:
        """
        Returns the union of the parameters and buffers of the module.
        """

        res = dict()
        res.update(self.parameter_dict)
        res.update(self.buffer_dict)

        return res

    def register_prior(self, name, prior):
        """
        Registers a ``pyfilter.distributions.Prior`` object together with a ``pyfilter.PriorBoundParameter`` on self.

        Args:
            name: The name to use for the prior.
            prior: The prior of the parameter.
        """

        self.prior_dict[name] = prior
        self.parameter_dict[name] = PriorBoundParameter(prior().sample(), requires_grad=False)

    def parameters_and_priors(self) -> Iterable[Tuple[PriorBoundParameter, "Prior"]]:
        """
        Returns the priors and parameters of the module as an iterable of tuples, i.e::
            [(parameter_0, prior_parameter_0), ..., (prior_parameter_n, prior_parameter_n)]
        """

        for prior, parameter in zip(self.prior_dict.values(), self.parameter_dict.values()):
            yield parameter, prior

        for module in filter(lambda u: isinstance(u, HasPriorsModule), self.children()):
            for prior, parameter in module.parameters_and_priors():
                yield prior, parameter

    def priors(self) -> Iterable["Prior"]:
        """
        Same as ``.parameters_and_priors()`` but only returns the priors.
        """

        for _, prior in self.parameters_and_priors():
            yield prior

    def sample_params(self, shape: torch.Size = torch.Size([])):
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

        Args:
            constrained: Optional parameter specifying whether to evaluate the original prior, or the bijected prior.
        """

        return sum((prior.eval_prior(p, constrained) for p, prior in self.parameters_and_priors()))

    def concat_parameters(self, constrained=False) -> torch.Tensor:
        """
        Concatenates the parameters into one tensor.

        Args:
            constrained: Optional parameter specifying whether to concatenate the original parameters, or bijected.
        """

        res = tuple(
            (p if constrained else prior.get_unconstrained(p)).view(-1, prior.get_numel(constrained))
            for p, prior in self.parameters_and_priors()
        )

        return torch.cat(res, dim=-1)

    def update_parameters_from_tensor(self, x: torch.Tensor, constrained=False):
        """
        Update the parameters of ``self`` with the last dimension of ``x``.

        Args:
            x: The tensor containing the new parameter values.
            constrained: Optional parameter indicating whether values in ``x`` are considered constrained to the
                parameters' original space.

        Example:
            >>> from pyfilter import timeseries as ts, distributions as dists
            >>> from torch.distributions import Normal, Uniform
            >>> import torch
            >>>
            >>> alpha_prior = dists.Prior(Normal, loc=0.0, scale=1.0)
            >>> beta_prior = dists.Prior(Uniform, low=-1.0, high=1.0)
            >>>
            >>> ar = ts.models.AR(alpha_prior, beta_prior, 0.05)
            >>> ar.sample_params(torch.Size([1]))
            >>>
            >>> new_values = torch.empty(2).normal_()
            >>> ar.update_parameters_from_tensor(new_values, constrained=False)
            >>> assert (new_values == ar.concat_parameters(constrained=False)).all()
        """

        expected_shape = self.concat_parameters(constrained=constrained)
        if x.shape[-1] != expected_shape.shape[-1]:
            raise Exception(f"Shapes not congruent! Expected {expected_shape}, got {x.shape}")

        left_index = 0
        for p, prior in self.parameters_and_priors():
            right_index = left_index + prior.get_numel(constrained=constrained)

            p.update_values(x[..., left_index:right_index], prior, constrained=constrained)
            left_index = right_index
