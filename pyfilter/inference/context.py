import threading
from collections import OrderedDict
from typing import Iterable, Tuple, List, Dict, Any, OrderedDict as tOrderedDict, Callable
import torch

from .prior import Prior
from .parameter import PriorBoundParameter
from .qmc import QuasiRegistry


class NotSamePriorError(Exception):
    pass


class ParameterDoesNotExist(Exception):
    pass


# TODO: Consider inheriting from torch.nn.Module
class InferenceContext(object):
    """
    Defines a parameter context in which we define parameters and priors.
    """

    _PARAMETER_KEY = "parameters"
    _PRIOR_KEY = "prior"

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
    def stack(self) -> List["InferenceContext"]:
        return self.__class__._contexts.stack

    def __enter__(self):
        self.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.remove(self)

        if exc_val:
            raise exc_val

    @classmethod
    def get_context(cls) -> "InferenceContext":
        """
        Returns the latest context.
        """

        if any(cls._contexts.stack):
            return cls._contexts.stack[-1]

        raise Exception(f"There are currently no active '{InferenceContext.__name__}'!")

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
            if self._prior_dict[name].equivalent_to(prior):
                return self.get_parameter(name)
            
            raise NotSamePriorError(f"You are trying to register a parameter for '{name}' that already exists, but the priors don't match!")

        self._prior_dict[name] = prior

        dist = prior.build_distribution()
        assert dist.batch_shape == torch.Size([]), "You cannot pass a batched distribution!"

        v = dist.sample()

        # TODO: Set name and context on init...
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

        if name in self._parameter_dict:
            return self._parameter_dict[name]
        
        raise ParameterDoesNotExist(f"No such parameter '{name}'!")

    def get_parameters(self, constrained=True) -> Iterable[Tuple[str, PriorBoundParameter]]:
        """
        Returns an iterable of the parameters.
        """

        for k, v in self._parameter_dict.items():
            yield k, (v if constrained else v.get_unconstrained())

    def stack_parameters(self, constrained=True) -> torch.Tensor:
        """
        Stacks the parameters such that we get a two-dimensional array where the first dimension corresponds to the
        total batch shape, and the last corresponds to the stacked and flattened samples of the distributions.

        Args:
             constrained: whether to stack the constrained or unconstrained parameters.
        """

        res = tuple()
        shape_dict = self._shape_dict if constrained else self._unconstrained_shape_dict
        
        for n, p in self.get_parameters():
            shape = shape_dict[n]
            v = (p if constrained else p.get_unconstrained()).view(-1, shape.numel())
            res += (v,)

        return torch.cat(res, dim=-1)

    def _apply_to_params(self, x: torch.Tensor, shape_dict, f):
        r"""
        Utility method.
        """

        index = 0
        for n, p in self.get_parameters():
            numel = shape_dict[n].numel()

            param = x[..., index: index + numel]
            f(p, param)

            index += numel

    def unstack_parameters(self, x: torch.Tensor, constrained=True):
        """
        Un-stacks and updates parameters given the :class:`torch.Tensor` ``x``.

        Args:
            x: the tensor to unstack and use for updating.
            constrained: whether the values of ``x`` are considered constrained.
        """

        shape_dict = self._shape_dict if constrained else self._unconstrained_shape_dict
        tot_len = sum(s.numel() for s in shape_dict.values())

        assert tot_len == x.shape[-1], "Total length of parameters is different from parameters in context!"

        self._apply_to_params(x, shape_dict, lambda u, v: u.update_values_(v, constrained=constrained))

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

    def exchange(self, other: "InferenceContext", mask: torch.BoolTensor):
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

    def make_new(self) -> "InferenceContext":
        """
        Creates a new context.
        """

        return InferenceContext()

    def state_dict(self) -> tOrderedDict[str, Any]:
        """
        Returns the state dictionary of ``self``.
        """

        res = OrderedDict([])
        res[self._PARAMETER_KEY] = {k: v.data for k, v in self.parameters.items()}
        res[self._PRIOR_KEY] = {k: v.state_dict() for k, v in self._prior_dict.items()}

        return res

    def load_state_dict(self, state_dict: tOrderedDict[str, Any]):
        """
        Loads the state dict from other context. Note that this method only verifies that the parameters of the priors
        are same as when saving the initial state, it does not compare the actual distribution.

        Args:
            state_dict: state of other context.
        """

        assert set(self.parameters.keys()) == set(state_dict[self._PARAMETER_KEY].keys())

        for k, v in self._prior_dict.items():
            for p_name, p_value in v._get_parameters().items():
                msg = f"Seems that you don't have the same parameters for '{p_name}'!"
                assert (p_value == state_dict[self._PRIOR_KEY][k][p_name]).all(), msg

            p = self.get_parameter(k)
            p.data = state_dict[self._PARAMETER_KEY][k]

        return

    def apply_fun(self, f: Callable[[PriorBoundParameter], torch.Tensor]) -> "InferenceContext":
        """
        Applies ``f`` to each parameter of ``self`` and returns a new :class:`ParameterContext`.

        Args:
            f: function to apply.
        """

        new_context = self.make_new()
        
        for k, v in self._prior_dict.items():
            p = new_context.named_parameter(k, v.copy().cpu())
            p.data = f(self.get_parameter(k).clone()).cpu()

        return new_context

    def copy(self):
        r"""
        Performs a copy of the current context.
        """

        return self.apply_fun(lambda p: p)


# TODO: Figure out whether you need to save the QMC state in the state dict?
class QuasiParameterContext(InferenceContext):
    r"""
    Implements a parameter context for quasi random sampling.
    """

    def __init__(self, randomize: bool = True):
        """
        Initializes the :class:`QuasiParameterContext` class.
        """

        super(QuasiParameterContext, self).__init__()
        self._quasi_key: int = None
        self._randomize = randomize

    def initialize_parameters(self, batch_shape: torch.Size):
        # NB: We use the un-constrained shape as that is what all algorithms use
        out = self.stack_parameters(constrained=False)

        self._quasi_key = QuasiRegistry.add_engine(out.shape[-1], self._randomize)
        probs = QuasiRegistry.sample(self._quasi_key, batch_shape).to(out.device)

        self._apply_to_params(
            probs,
            self._unconstrained_shape_dict,
            lambda u, v: u.inverse_sample_(v.view(batch_shape + u.prior().event_shape), constrained=False)
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.stack) == 1:
            QuasiRegistry.remove_engine(self._quasi_key)

        super(QuasiParameterContext, self).__exit__(exc_type, exc_val, exc_tb)

    def make_new(self) -> "InferenceContext":
        return QuasiParameterContext(randomize=self._randomize)


def make_context(use_quasi: bool = False, randomize: bool = True) -> InferenceContext:
    """
    Helper method for creating a context.
    """

    if use_quasi:
        return QuasiParameterContext(randomize=randomize)

    return InferenceContext()
