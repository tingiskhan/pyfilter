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

        self._shape_dict: Dict[str, torch.Size] = OrderedDict([])
        self._unconstrained_shape_dict: Dict[str, torch.Size] = OrderedDict([])

    @property
    def parameters(self) -> Dict[str, PriorBoundParameter]:
        """
        Returns the parameters.
        """

        return self._parameter_dict

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

        return self._prior_dict.get(name, None)

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
            # TODO: Add check for whether prior is the same or not, if not then raise
            return self.get_parameter(name)

        assert self in self.stack, "Cannot register parameters in an inactive context!"

        self._prior_dict[name] = prior

        v = prior.build_distribution().sample()

        self._parameter_dict[name] = parameter = PriorBoundParameter(v, requires_grad=False)
        parameter.set_context(self)
        parameter.set_name(name)

        self._shape_dict[name] = prior.build_distribution().event_shape
        self._unconstrained_shape_dict[name] = prior.unconstrained_prior.event_shape

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

    def stack_parameters(self, constrained=True, dim=-1) -> torch.Tensor:
        """
        Stacks the parameters along ``dim`` in order of registration.
        """

        res = tuple()
        shape_dict = self._shape_dict if constrained else self._unconstrained_shape_dict
        for n, p in self.get_parameters():
            shape = shape_dict[n]

            event_dim = len(shape)
            v = (p if constrained else p.get_unconstrained()).flatten(end_dim=-(event_dim + 1))

            res += (v if dim > 0 else v.unsqueeze(-1),)

        return torch.cat(res, dim=dim)

    def unstack_parameters(self, x: torch.Tensor, constrained=True):
        """
        Unstacks and updates parameters given the :class:`torch.Tensor` ``x``.

        Args:
            x: the tensor to unstack and use for updating.
            constrained: whether the values of ``x`` are considered constrained.
        """

        shape_dict = self._shape_dict if constrained else self._unconstrained_shape_dict
        tot_len = sum(s.numel() for s in shape_dict.values())

        assert tot_len == x.shape[-1], f"Total length of parameters is different from parameters in context!"

        index = 0
        for n, p in self.get_parameters():
            numel = shape_dict[n].numel()

            param = x[..., index:index + numel]
            p.update_values_(param, constrained=constrained)

            index += numel

    def initialize_parameters(self, batch_shape: torch.Size):
        """
        Initializes the parameters by sampling from the priors.

        Args:
            batch_shape: the batch shape to use.
        """

        for _, p in self.get_parameters():
            p.sample_(batch_shape)

    def eval_priors(self, constrained=True) -> torch.Tensor:
        """
        Evaluates the priors.

        Args:
            constrained: whether to evaluate the constrained parameters.

        """

        return sum(p.eval_prior(constrained=constrained) for _, p in self.get_parameters())

    def exchange(self, other: "ParameterContext", mask: torch.BoolTensor):
        """
        Exchanges the parameters of ``self`` with that of ``other``.

        Args:
            other: the :class:`ParameterContext` to take the values from.
            mask: the mask from where to take.
        """

        for n, p in self.get_parameters():
            other_p = other.get_parameter(n)
            # TODO: Change when masked_scatter works
            p[mask] = other_p[mask]

    def resample(self, indices: torch.IntTensor):
        """
        Resamples the parameters of ``self`` given ``indices``.

        Args:
            indices: the indices at which to resample.

        """

        for n, p in self.get_parameters():
            p.copy_(p[indices])

    @classmethod
    def make_new(cls) -> "ParameterContext":
        """
        Creates a new context.
        """

        return ParameterContext()


def make_context() -> ParameterContext:
    """
    Helper method for creating a context.
    """

    return ParameterContext.make_new()
