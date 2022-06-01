from pyro.distributions import Normal
import torch
from torch.autograd import grad
from .random_walk import RandomWalk


class GradientBasedProposal(RandomWalk):
    r"""
    Implements a proposal kernel that utilizes the gradient of the total log likelihood, which we define as
        .. math::
            S(\theta) \coloneqq \log{p_\theta(y_{1:t})} + \log{p(\\theta)}.

    The candidate kernel :math:`\theta^*` is then generated by
        .. math::
            \theta^* \sim \mathcal{N} \left (\theta + \epsilon \nabla S(\theta), \sqrt{2\epsilon} \right),

    where :math:`\\theta` denotes the latest accepted parameter candidate, and :math:`\\epsilon` the step size. Note
    that we generate the kernels on the constrained space of the parameters.
    """

    def __init__(self, use_second_order: bool = False, **kwargs):
        """
        Initializes the ``GradientBasedProposal`` class.

        Args:
            use_second_order: optional parameter specifying whether to use second order information when constructing
                the proposal kernel. In practice this means that we utilize the diagonal of the Hessian.
            kwargs: see base.
        """

        super().__init__(**kwargs)
        self._eps = self._scale ** 2.0 / 2.0
        self._use_second_order = use_second_order

    def build(self, context, state, filter_, y):
        smoothed = filter_.smooth(state.filter_state.states)

        params = context.stack_parameters(constrained=False)
        params.requires_grad_(True)

        context.unstack_parameters(params, constrained=False)

        time = torch.stack(tuple(s.x.time_index for s in state.filter_state.states))

        # As the first state's time value is zero, we use that
        first_state = state.filter_state.states[0].get_timeseries_state()

        xtm1 = first_state.propagate_from(values=smoothed[:-1], time_increment=time[:-1])
        xt = first_state.propagate_from(values=smoothed[1:], time_increment=time[1:])

        y = y.view(y.shape[0], 1, 1, *y.shape[1:])

        hidden_dens = filter_.ssm.hidden.build_density(xtm1)
        obs_dens = filter_.ssm.build_density(xt)

        logl = filter_.ssm.hidden.initial_dist.log_prob(smoothed[0]).mean(-1)
        logl += context.eval_priors(constrained=False).squeeze(-1)
        logl += (hidden_dens.log_prob(xt.values) + obs_dens.log_prob(y)).mean(-1).sum(0)

        g = grad(logl, params, torch.ones_like(logl), create_graph=self._use_second_order)[-1]

        ones = torch.ones_like(params)
        step = self._eps * ones
        scale = self._scale * ones

        if self._use_second_order:
            raise NotImplementedError("Second order information is currently not implemented!")

        loc = params.detach_() + step * g

        for _, v in context.get_parameters():
            v.detach_()

        return Normal(loc=loc, scale=scale).to_event(1)
