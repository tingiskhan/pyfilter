import torch
from pyro.distributions import Normal
from torch.autograd import grad

from .random_walk import RandomWalk


class GradientBasedProposal(RandomWalk):
    r"""
    Implements a proposal kernel that utilizes the gradient of the total log likelihood, which we define as
        .. math::
            S(\theta) \coloneqq \log{p_\theta(y_{1:t})} + \log{p(\theta)}.

    The candidate kernel :math:`\theta^*` is then generated by
        .. math::
            \theta^* \sim \mathcal{N} \left (\theta + \epsilon \nabla S(\theta), \sqrt{2\epsilon} \right),

    where :math:`\theta` denotes the latest accepted parameter candidate, and :math:`\epsilon` the step size. Note
    that we generate the kernels on the constrained space of the parameters.
    """

    def __init__(self, use_second_order: bool = False, **kwargs):
        """
        Internal initializer for :class:`GradientBasedProposal`.

        Args:
            use_second_order (bool, optional): whether to use seconrd order information when constructing proposal
            kernel. Defaults to False.
        """

        if use_second_order:
            raise NotImplementedError("Currently does not support `use_second_order`!")

        super().__init__(**kwargs)
        self._eps = self._scale ** 2.0 / 2.0
        self._use_second_order = use_second_order

    # TODO: Use functorch...
    def build(self, context, state, filter_, y):
        smoothed = filter_.smooth(state.filter_state.states)

        params = context.stack_parameters(constrained=False)
        params.requires_grad_(True)

        context.unstack_parameters(params, constrained=False)
        
        with context.no_prior_verification():
            filter_.initialize_model(context)

        time = torch.stack([s.timeseries_state.time_index for s in state.filter_state.states])

        # As the first state's time value is zero, we use that
        first_state = state.filter_state.states[0].get_timeseries_state()

        xtm1 = first_state.propagate_from(values=smoothed[:-1], time_increment=time[:-1])
        x_t = first_state.propagate_from(values=smoothed[1:], time_increment=time[1:])
        
        hidden_dens = filter_.ssm.hidden.build_density(xtm1)
        obs_dens = filter_.ssm.build_density(x_t)

        y = y.reshape(y.shape[:1] + torch.Size([1 for _ in hidden_dens.batch_shape[1:]]) + obs_dens.event_shape)

        logl = filter_.ssm.hidden.initial_distribution.log_prob(smoothed[0]).mean(0)
        logl += context.eval_priors(constrained=False)
        logl += (hidden_dens.log_prob(x_t.value) + obs_dens.log_prob(y)).mean(0).sum(0)

        gradient = grad(logl, params, torch.ones_like(logl), create_graph=self._use_second_order)[-1]

        step = torch.full_like(params, self._eps)
        scale = torch.full_like(params, self._scale)

        if self._use_second_order:
            raise NotImplementedError("Second order information is currently not implemented!")

        loc = params.detach_() + step * gradient

        for _, v in context.get_parameters():
            v.detach_()

        return Normal(loc=loc, scale=scale).to_event(1)
