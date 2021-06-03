import torch
from torch.distributions import Distribution
from . import JointState
from .stochasticprocess import StochasticProcess
from .affine import AffineProcess
from ..distributions import JointDistribution, DistributionWrapper


class JointStochasticProcess(StochasticProcess):
    """
    Implements an object for combining multiple series into one by assuming independence.
    """

    def __init__(self, **processes: StochasticProcess):
        if any(isinstance(p, JointStochasticProcess) for p in processes.values()):
            raise NotImplementedError("Currently does not handle joint of joint!")

        # TODO: Feels kinda hacky?
        processes = {k: v for k, v in processes.items() if isinstance(v, StochasticProcess)}

        joint_dist = JointDistribution(*(p.initial_dist for p in processes.values()))
        super().__init__(DistributionWrapper(lambda **u: joint_dist))

        self.masks = joint_dist.masks
        self._proc_names = tuple(processes.keys())
        for name, proc in processes.items():
            self.add_module(name, proc)

    def initial_sample(self, shape=None) -> JointState:
        return JointState.from_states(
            *(self._modules[name].initial_sample(shape) for name in self._proc_names),
            mask=self.masks
        )

    def build_density(self, x: JointState) -> Distribution:
        return JointDistribution(
            *(self._modules[name].build_density(x[i]) for i, name in enumerate(self._proc_names)),
            masks=self.masks
        )


class AffineJointStochasticProcesses(AffineProcess, JointStochasticProcess):
    """
    Implements a joint process where all sub processes are of affine nature.
    """

    def __init__(self, **processes: AffineProcess):
        if not all(isinstance(p, AffineProcess) for p in processes.values()):
            raise ValueError(f"All processes must be of type '{AffineProcess.__name__}'!")

        super(AffineJointStochasticProcesses, self).__init__(
            (None, None), (), increment_dist=None, initial_dist=None, **processes
        )

        self.increment_dist = DistributionWrapper(
            lambda **u: JointDistribution(*(self._modules[name].increment_dist() for name in self._proc_names))
        )

    def mean_scale(self, x: JointState, parameters=None):
        mean = tuple()
        scale = tuple()

        for i, (proc_name, p) in enumerate(zip(self._proc_names, parameters or len(self._proc_names) * [None])):
            proc: AffineProcess = self._modules[proc_name]
            m, s = torch.broadcast_tensors(*proc.mean_scale(x[i], p))

            mean += (m.unsqueeze(-1) if proc.n_dim == 0 else m,)
            scale += (s.unsqueeze(-1) if proc.n_dim == 0 else s,)

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-1)

     # TODO: Should perhaps return a flat list which is split later on, but I think this is better
    def functional_parameters(self, **kwargs):
        return tuple((self._modules[proc_name].functional_parameters(**kwargs) for proc_name in self._proc_names))
