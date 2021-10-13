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

        self._priors_dict = ModuleDict()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._parameter_dict = ParameterDict()
            self._buffer_dict = BufferDict()

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
            self._parameter_dict.update({name: p})
        else:
            self._buffer_dict.update({name: p if (isinstance(p, torch.Tensor) or p is None) else torch.tensor(p)})

    def get_parameters_and_buffers(self) -> Dict[str, Union[torch.Tensor, torch.nn.Parameter]]:
        """
        Returns the union of the parameters and buffers of the module.
        """

        res = dict()
        res.update(self._parameter_dict)
        res.update(self._buffer_dict)

        return res

    def register_prior(self, name, prior):
        """
        Registers a ``pyfilter.distributions.Prior`` object together with a ``pyfilter.PriorBoundParameter`` on self.

        Args:
            name: The name to use for the prior.
            prior: The prior of the parameter.
        """

        self._priors_dict[name] = prior
        self._parameter_dict.update({name: PriorBoundParameter(prior().sample(), requires_grad=False)})

    def parameters_and_priors(self) -> Iterable[Tuple["Prior", PriorBoundParameter]]:
        """
        Returns the priors and parameters of the module as an iterable of tuples, i.e::
            [(prior_parameter_0, parameter_0), ..., (prior_parameter_n, parameter_n)]
        """

        for prior, parameter in zip(self._priors_dict.values(), self._parameter_dict.values()):
            yield prior, parameter

        for module in filter(lambda u: isinstance(u, HasPriorsModule), self.children()):
            yield module.parameters_and_priors()

    def priors(self) -> Iterable["Prior"]:
        """
        Same as ``.parameters_and_priors()`` but only returns the priors.
        """

        for prior, _ in self.parameters_and_priors():
            yield prior

    def sample_params(self, shape: torch.Size):
        """
        Samples the parameters of the model in place.

        Args:
            shape: The shape of the parameters to use when sampling.
        """

        for param, prior in self.parameters_and_priors():
            param.sample_(prior, shape)

        return self
