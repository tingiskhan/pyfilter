from .normalization import normalize
import torch
from torch.distributions import Distribution
from math import sqrt
from typing import Union, Tuple, Iterable


EPS = sqrt(torch.finfo(torch.float32).eps)


def get_ess(w: torch.Tensor, normalized=False):
    """
    Calculates the ESS from an array of log weights.
    :param w: The log weights
    :param normalized: Whether input is normalized
    :return: The effective sample size
    """

    if not normalized:
        w = normalize(w)

    return w.sum(-1) ** 2 / (w ** 2).sum(-1)


def choose(array: torch.Tensor, indices: torch.Tensor):
    """
    Function for choosing on either columns or index.
    :param array: The array to choose on
    :param indices: The indices to choose from `array`
    :return: Returns the chosen elements from `array`
    """

    if indices.dim() < 2:
        return array[indices]

    return array[torch.arange(array.shape[0], device=array.device)[:, None], indices]


def loglikelihood(w: torch.Tensor, weights: torch.Tensor = None):
    """
    Calculates the estimated loglikehood given weights.
    :param w: The log weights, corresponding to likelihood
    :param weights: Whether to weight the log-likelihood.
    :return: The log-likelihood
    """

    maxw, _ = w.max(-1)

    # ===== Calculate the second term ===== #
    if weights is None:
        temp = (
            torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw))
            .mean(-1)
            .log()
        )
    else:
        temp = (
            (weights * torch.exp(w - (maxw.unsqueeze(-1) if maxw.dim() > 0 else maxw)))
            .sum(-1)
            .log()
        )

    return maxw + temp


def concater(*x: Union[Iterable[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Concatenates output.
    :type x: tuple[torch.Tensor]|torch.Tensor
    """

    if isinstance(x, torch.Tensor):
        return x

    return torch.stack(torch.broadcast_tensors(*x), dim=-1)


def construct_diag(x: torch.Tensor):
    """
    Constructs a diagonal matrix based on batched data. Solution found here:
    https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p
    Do note that it only considers the last axis.
    :param x: The tensor
    """

    if x.dim() < 1:
        return x
    elif x.shape[-1] < 2:
        return x.unsqueeze(-1)
    elif x.dim() < 2:
        return torch.diag(x)

    b = torch.eye(x.size(-1), device=x.device)
    c = x.unsqueeze(-1).expand(*x.size(), x.size(-1))

    return c * b


def flatten(*args: Iterable[Iterable]) -> Tuple:
    """
    Flattens an array comprised of an arbitrary number of lists. Solution found at:
        https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    :param args: The iterable you wish to flatten.
    :type args: collections.Iterable
    :return: Flattened iterable
    """
    out = list()
    for el in args:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes, torch.Tensor)):
            out.extend(flatten(*el))
        else:
            out.append(el)

    return tuple(out)


def unflattify(values: torch.Tensor, shape: torch.Size):
    """
    Unflattifies parameter values.
    :param values: The flattened array of values that are to be unflattified
    :param shape: The shape of the parameter prior
    """

    if len(shape) < 1 or values.shape[1:] == shape:
        return values

    return values.reshape(values.shape[0], *shape)


class StackedObject(object):
    def __init__(self, concated: torch.Tensor, mask: torch.Tensor, prev_shape: torch.Size):
        """
        Object for storing the results of stacking tensors from `stacker`.
        """

        self.concated = concated
        self.mask = mask
        self.prev_shape = prev_shape


def stacker(parameters: Iterable[torch.Tensor], selector=lambda u: u.values, dim=1):
    """
    Stacks the parameters and returns a n-tuple containing the mask for each parameter.
    :param parameters: The parameters
    :type parameters: tuple[Parameter]|list[Parameter]
    :param selector: The selector
    :param dim: The dimension to start flattening from
    """

    to_conc = tuple()
    mask = tuple()
    prev_shape = tuple()

    i = 0
    # TODO: Currently only supports one sampling dimension...
    for p in parameters:
        s = selector(p)
        flat = s if s.dim() <= dim else s.flatten(dim)

        if flat.dim() == dim:
            to_conc += (flat.unsqueeze(-1),)
            slc = i
        else:
            to_conc += (flat,)
            slc = slice(i, i + flat.shape[-1])

        mask += (slc,)
        i += to_conc[-1].shape[-1]
        prev_shape += (s.shape[dim:],)

    return StackedObject(torch.cat(to_conc, dim=-1), mask, prev_shape)


class TempOverride(object):
    def __init__(self, obj: object, attr: str, new_vals: object):
        """
        Implements a temporary override of attribute of an object.
        :param obj: An object
        :param attr: The attribute to override
        :param new_vals: The new values
        """
        self._obj = obj
        self._attr = attr
        self._new_vals = new_vals
        self._old_vals = None

    def __enter__(self):
        self._old_vals = getattr(self._obj, self._attr)
        setattr(self._obj, self._attr, self._new_vals)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(self._obj, self._attr, self._old_vals)

        return False


class Empirical(Distribution):
    def __init__(self, samples: torch.Tensor):
        """
        Helper class for timeseries without an analytical expression.
        :param samples: The sample
        """
        super().__init__()
        self.loc = self._samples = samples
        self.scale = torch.zeros_like(samples)

    def sample(self, sample_shape=torch.Size()):
        if sample_shape != self._samples.shape and sample_shape != torch.Size():
            raise ValueError('Current implementation only allows passing an empty size!')

        return self._samples