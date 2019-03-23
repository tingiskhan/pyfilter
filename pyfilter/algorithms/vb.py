from .base import BatchAlgorithm
import torch
import torch.distributions as dists
from torch import optim
import tqdm
from ..proposals.linearized import eps


class VBResultSet(object):
    def __init__(self, final_elbo, mean, logstd):
        """
        Naive results from VariationalBayes.
        :param final_elbo: The final elbo
        :type final_elbo: float
        :param mean: The mean vector of the state
        :type mean: torch.Tensor
        :param logstd: The log std vector of the state
        :type logstd: torch.Tensor
        """

        self.final_elbo = final_elbo
        self.mean = mean
        self.logstd = logstd


class VariationalBayes(BatchAlgorithm):
    def __init__(self, model, num_samples=4, fullrank=False, optimizer=optim.Adam, maxiters=30e3, optkwargs=None):
        """
        Implements Variational Bayes.
        :param model: The model
        :type model: pyfilter.timeseries.StateSpaceModel
        :param num_samples: The number of samples
        :type num_samples: int
        :param fullrank: Whether to use full rank approximation
        :type fullrank: bool
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
        self._fullrank = fullrank

        self._weightmat = torch.ones((self._numsamples, self._model.hidden_ndim))
        self._optimizer = optimizer
        self._maxiters = int(maxiters)
        self.optkwargs = optkwargs or dict()
        self._resultset = None

    @property
    def result_set(self):
        """
        Returns the result set of the fitting procedure
        :rtype: VBResultSet
        """

        return self._resultset

    def loss(self, y, eta, mean, logstd):
        """
        The loss function, i.e. loglikelihood/ELBO
        :rtype: torch.Tensor
        """

        # ===== Sample ===== #
        samples = eta.sample((self._numsamples,))

        if not self._fullrank:
            scale = logstd.exp()
            transformed = mean + scale * samples

            entropydist = dists.Independent(dists.Normal(mean, scale), 2)
        else:
            raise NotImplementedError('Full-rank is currently not implemented!')

        # ===== Helpers ===== #
        x_t = transformed[:, 1:]
        x_tm1 = transformed[:, :-1]

        # ===== Loss function ===== #
        logl = (self._model.weight(y, x_t) + self._model.h_weight(x_t, x_tm1)).sum(1).mean(0)

        return -(logl + entropydist.entropy())

    def _fit(self, y):
        # ===== Get shape of state vectors ====== #
        mean = torch.zeros((y.shape[0] + 1, self._model.hidden_ndim), requires_grad=True)

        if not self._fullrank:
            logstd = torch.ones_like(mean, requires_grad=True)
        else:
            raise NotImplementedError('Full-rank is currently not implemented!')

        # ===== Start optimization ===== #
        eta = dists.Independent(dists.Normal(torch.zeros_like(mean), torch.ones_like(logstd)), 2)

        optimizer = self._optimizer([mean, logstd], **self.optkwargs)
        elbo_old = torch.tensor(1e6)
        elbo = torch.tensor(12e6)

        it = 0
        bar = tqdm.tqdm(total=self._maxiters)
        while (elbo - elbo_old).abs() > eps and it < self._maxiters:
            elbo_old = elbo

            # ===== Perform optimization ===== #
            optimizer.zero_grad()
            elbo = self.loss(y, eta, mean, logstd)
            elbo.backward()
            optimizer.step()

            it += 1
            bar.update(1)
            bar.set_description('{:s} - ELBO: {:.2f}'.format(str(self), -elbo[-1]))

        # ===== Define result set ===== #
        self._resultset = VBResultSet(elbo.detach()[-1], mean.detach(), logstd.detach())

        return self
