import torch
from .utils import run_pmmh
from .proposals import RandomWalk, BaseProposal
from .state import PMMHResult
from ...context import make_context
from ...base import BaseAlgorithm
from ...logging import TQDMWrapper


class PMMH(BaseAlgorithm):
    """
    Implements the `Particle Marginal Metropolis Hastings` algorithm found in `Particle Markov chain Monte Carlo
    methods`_ by C. Andrieu et al.

    .. _`Particle Markov chain Monte Carlo methods`: https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf
    """

    MONTE_CARLO_SAMPLES = torch.Size([10_000])

    def __init__(
        self, filter_, num_samples: int, num_chains: int = 4, proposal: BaseProposal = None, initializer: str = "mean", context=None
    ):
        r"""
        Initializes the :class:`PMMH` class.

        Args:
             filter_: see base.
             num_samples: the number of PMMH samples to draw.
             num_chains: the number of parallel chains to run. The total number of samples on termination is thus
                ``samples * num_chains``. Do note that we utilize broadcasting rather than ``for``-loops.
             proposal: optional parameter specifying how to construct the proposal density for candidate
                :math:`\theta^*` given the previously accepted candidate :math:`\theta_i`. If not specified, defaults
                to :class:`pyfilter.inference.batch.mcmc.proposals.RandomWalk`.
            initializer: Optional parameter specifying how to initialize the chain:
                - ``seed``: Seeds the initial value by running several chains in parallel and choosing the one
                    maximizing the total likelihood
                - ``mean``: Sets the initial values as the means of the prior distributions. Uses MC sampling for
                    determining the mean.
        """

        super().__init__(filter_, context=context)
        
        self.num_samples = num_samples
        self._num_chains = torch.Size([num_chains])
        self._parameter_shape = torch.Size([num_chains, 1])

        self.context.set_batch_shape(self._parameter_shape)
        self.filter.set_batch_shape(self._num_chains)

        self._proposal = proposal or RandomWalk()
        self._initializer = initializer

    def initialize(self, y: torch.Tensor) -> PMMHResult:
        self.filter.initialize_model(self.context)
        
        if self._initializer == "seed":
            raise NotImplementedError()
        elif self._initializer == "mean":
            for p in self.filter.ssm.parameters():
                dist = p.prior.build_distribution()
                mean = dist.sample(self.MONTE_CARLO_SAMPLES).mean(dim=0)

                p.data = mean * torch.ones(self._parameter_shape, device=p.device)
        else:
            raise NotImplementedError(f"``{self._initializer}`` is not configured!")

        prev_res = self._filter.batch_filter(y, bar=False)

        return PMMHResult(dict(self.context.get_parameters()), prev_res)

    def fit(self, y: torch.Tensor, logging=None, **kwargs):
        state = self.initialize(y)

        logging = logging or TQDMWrapper()

        with logging.initialize(self, self.num_samples):
            prop_dist = self._proposal.build(self.context, state, self._filter, y)

            with self.context.make_new() as sub_context:
                proposal_filter = self.filter.copy()
                sub_context.set_batch_shape(self._parameter_shape)
                proposal_filter.initialize_model(sub_context)

            for i in range(self.num_samples):
                accepted = run_pmmh(
                    self.context,
                    state,
                    self._proposal,
                    prop_dist,
                    proposal_filter,
                    sub_context,
                    y,
                    mutate_kernel=True,
                )

                state.update_chain(dict(self.context.get_parameters()))
                logging.do_log(i, state)

            return state
