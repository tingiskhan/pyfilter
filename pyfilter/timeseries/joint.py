import torch
from torch.distributions import Distribution
from . import JointState
from .stochasticprocess import StochasticProcess
from .affine import AffineProcess
from ..distributions import JointDistribution, DistributionWrapper


class JointStochasticProcess(StochasticProcess):
    """
    A stochastic process comprising multiple separate stochastic processes by assuming independence between them. That
    is, given :math:`n` stochastic processes :math:`\{X^i_t\}, i = 1, \dots, n\:`
        .. math::
            p(x^1_{t+1}, \dots, x^n_{t+1} \mid x^1_t, \dots, x^n_t) = \prod^n_{i=1} p(x^i_{t+1} \mid x^i_t)

    Example:
        In this example we'll construct a joint process of a random walk and an Ornstein-Uhlenbeck process.
            >>> from pyfilter.timeseries import models as m, JointStochasticProcess
            >>>
            >>> ou = m.OrnsteinUhlenbeck(0.01, 0.0, 0.05, 1, 1.0)
            >>> rw = m.RandomWalk(0.05)
            >>>
            >>> joint = JointStochasticProcess(ou=ou, rw=rw)
            >>> x = joint.sample_path(1000)
            >>> x.shape
            torch.Size([1000, 2])
    """

    def __init__(self, **processes: StochasticProcess):
        """
        Initializes the ``JointStochasticProcess`` class.

        Args:
            processes: Key-worded processes, where the process will be registered as a module with key.
        """

        if any(isinstance(p, JointStochasticProcess) for p in processes.values()):
            raise NotImplementedError("Currently does not handle joint of joint!")

        # TODO: Feels kinda hacky?
        processes = {k: v for k, v in processes.items() if isinstance(v, StochasticProcess)}

        joint_dist = JointDistribution(*(p.initial_dist for p in processes.values()))
        super().__init__(DistributionWrapper(lambda **u: joint_dist))

        self.indices = joint_dist.indices
        self._proc_names = tuple(processes.keys())
        for name, proc in processes.items():
            self.add_module(name, proc)

    def initial_sample(self, shape=None) -> JointState:
        return JointState.from_states(
            *(self._modules[name].initial_sample(shape) for name in self._proc_names), indices=self.indices
        )

    def build_density(self, x: JointState) -> Distribution:
        return JointDistribution(
            *(self._modules[name].build_density(x[i]) for i, name in enumerate(self._proc_names)), indices=self.indices
        )


class AffineJointStochasticProcesses(AffineProcess, JointStochasticProcess):
    """
    Similar to ``JointStochasticProcess`` but with the exception that all processes are of type ``AffineProcess``.
    """

    def __init__(self, **processes: AffineProcess):
        """
        Initializes the ``AffineJointStochasticProcesses`` class.

        Args:
            processes: See base.
        """

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
