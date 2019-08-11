from .base import BatchAlgorithm
import torch
from torch import optim
import tqdm
from math import sqrt
from .varapprox import MeanField, BaseApproximation, ParameterApproximation


eps = sqrt(torch.finfo(torch.float32).eps)


class VariationalBayes(BatchAlgorithm):
    def __init__(self, model, num_samples=4, approx=MeanField(), optimizer=optim.Adam, maxiters=30e3, optkwargs=None):
        """
        Implements Variational Bayes.
        :param model: The model
        :type model: pyfilter.timeseries.StateSpaceModel
        :param num_samples: The number of samples
        :type num_samples: int
        :param approx: The variational approximation to use for the latent space
        :type approx: BaseApproximation
        :param optimizer: The optimizer
        :type optimizer: optim.Optimizer
        :param maxiters: The maximum number of iterations
        :type maxiters: int|float
        :param optkwargs: Any optimizer specific kwargs
        :type optkwargs: dict
        """

        super().__init__(None)
        self._model = model
        self._numsamples = num_samples

        self._approximation = approx
        self._p_approx = None   # type: ParameterApproximation

        self._optimizer = optimizer
        self._maxiters = int(maxiters)
        self.optkwargs = optkwargs or dict()

        self._runavg = 0
        self._decay = 0.975

    @property
    def approximation(self):
        """
        Returns the resulting variational approximation
        :rtype: BaseApproximation
        """

        return self._approximation

    def loss(self, y):
        """
        The loss function, i.e. ELBO
        :rtype: torch.Tensor
        """

        # ===== Sample states ===== #
        transformed = self._approximation.sample(self._numsamples)

        # TODO: Clean this up and make it work for matrices
        # ===== Sample parameters ===== #
        params = self._p_approx.sample(self._numsamples)
        for i, p in enumerate(self._model.flat_theta_dists):
            p.detach_()
            p[:] = p.bijection(params[..., i])

        self._model.viewify_params((self._numsamples, 1))

        # ===== Helpers ===== #
        x_t = transformed[:, 1:]
        x_tm1 = transformed[:, :-1]

        if self._model.hidden_ndim < 2:
            x_t.squeeze_(-1)
            x_tm1.squeeze_(-1)

        # ===== Loss function ===== #
        logl = (self._model.weight(y, x_t) + self._model.h_weight(x_t, x_tm1)).sum(1).mean(0)
        entropy = self._approximation.entropy() + self._p_approx.entropy()

        return -(logl + torch.mean(self._model.p_prior(transformed=True), dtype=logl.dtype) + entropy)

    def _fit(self, y):
        # ===== Sample model in place for a primitive version of initialization ===== #
        self._model.sample_params(self._numsamples)

        # ===== Initialize the state approximation ===== #
        self._approximation.initialize(y, self._model.hidden_ndim)

        # ===== Setup the parameter approximation ===== #
        self._p_approx = ParameterApproximation().initialize(self._model.flat_theta_dists)

        # ===== Define the optimizer ===== #
        parameters = [*self._approximation.get_parameters(), *self._p_approx.get_parameters()]
        optimizer = self._optimizer(parameters, **self.optkwargs)

        elbo_old = -torch.tensor(float('inf'))
        elbo = -elbo_old

        it = 0
        bar = tqdm.tqdm(total=self._maxiters)
        while (elbo - elbo_old).abs() > eps and it < self._maxiters:
            elbo_old = elbo

            # ===== Perform optimization ===== #
            optimizer.zero_grad()
            elbo = self.loss(y)
            elbo.backward()
            optimizer.step()

            it += 1
            bar.update(1)
            self._runavg = self._runavg * self._decay - elbo * (1 - self._decay)
            bar.set_description('{:s} - Avg. ELBO: {:.2f}'.format(str(self), self._runavg))

        return self
