import torch
from .joint import JointStochasticProcess, AffineJointStochasticProcess
from ..distributions import JointDistribution


class ChainedStochasticProcess(JointStochasticProcess):
    """
    Implements a stochastic process which constitutes multiple "chained" stochastic processes, s.t. each process is
    conditionally independent given the previous process.

    # TODO: math
    """

    def build_density(self, x):
        return JointDistribution(
            *(self._modules[name].build_density(x[: i + 1]) for i, name in enumerate(self._proc_names)),
            indices=self.indices
        )


class AffineChainedStochasticProcess(AffineJointStochasticProcess):
    """
    Similar to ``ChainedStochasticProcess``, but all sub-processes must be of affine type - similar to
    ``AffineJointStochasticProcess``.
    """

    def mean_scale(self, x, parameters=None):
        mean = tuple()
        scale = tuple()

        for i, (proc_name, p) in enumerate(zip(self._proc_names, parameters or len(self._proc_names) * [None])):
            proc = self._modules[proc_name]
            m, s = torch.broadcast_tensors(*proc.mean_scale(x[: i + 1], p))

            mean += (m.unsqueeze(-1) if proc.n_dim == 0 else m,)
            scale += (s.unsqueeze(-1) if proc.n_dim == 0 else s,)

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-1)
