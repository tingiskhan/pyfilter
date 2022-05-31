import threading
from collections import OrderedDict
from typing import Iterable, Tuple, List, Dict

import torch
from .prior import Prior
from .parameter import PriorBoundParameter


class ParameterContext(object):
    """
    Defines a parameter context in which we define parameters and priors.
    """

    # NB: Same approach as in PyMC3
    _contexts = threading.local()
    _contexts.stack = list()

    def __init__(self):
        """
        Initializes the :class:`ParameterContext` class.
        """

        self._prior_dict: Dict[str, Prior] = OrderedDict([])
        self._parameter_dict: Dict[str, PriorBoundParameter] = OrderedDict([])

    @property
    def stack(self) -> List["ParameterContext"]:
        return self.__class__._contexts.stack

    def __enter__(self):
        self.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.remove(self)

        if exc_val:
            raise exc_val

    @classmethod
    def get_context(cls) -> "ParameterContext":
        """
        Returns the latest context.
        """

        return cls._contexts.stack[-1]

    def get_prior(self, name: str) -> Prior:
        """
        Returns the prior given the name of the parameter.
        """

        return self._prior_dict[name]

    def named_parameter(self, name: str, prior: Prior) -> PriorBoundParameter:
        """
        Registers a prior on the global prior dictionary, and creates a corresponding parameter.

        Args:
            name: name of the prior and parameter to register.
            prior: prior object.

        Returns:
            Returns a :class:`PriorBoundParameter`.
        """

        if name in self._prior_dict:
            raise KeyError(f"A prior with the key '{name}' already exists!")

        self._prior_dict[name] = prior

        v = prior.build_distribution().sample()

        self._parameter_dict[name] = parameter = PriorBoundParameter(v, requires_grad=False)
        parameter.set_context(self.get_context())
        parameter.set_name(name)

        return parameter

    def get_parameter(self, name: str) -> PriorBoundParameter:
        """
        Gets the parameter named ``name``.

        Args:
            name: name of the parameter.

        Returns:
            Returns the prior.
        """

        return self._parameter_dict.get(name, None)

    def get_parameters(self, constrained=True) -> Iterable[Tuple[str, PriorBoundParameter]]:
        """
        Returns an iterable of the parameters.
        """

        for k, v in self._parameter_dict.items():
            yield k, (v if constrained else v.get_unconstrained())

    def stack_parameters(self, constrained=True, dim=-1):
        """
        Stacks the parameters along `dim`.
        """

        res = tuple()
        for n, p in self.get_parameters():
            prior = p.prior.build_distribution()
            v = p if constrained else p.get_unconstrained()

            res += (v if prior.event_shape.numel() > 1 else v.unsqueeze(dim),)

        return torch.cat(res, dim=dim)


def make_context() -> ParameterContext:
    """
    Helper method for creating a context.
    """

    return ParameterContext()
