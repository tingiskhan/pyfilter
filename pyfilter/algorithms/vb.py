from .base import BatchAlgorithm
import torch
import torch.distributions as dists


class VariationalBayes(BatchAlgorithm):
    def __init__(self, model, num_samples=16, fullrank=False):
        """
        Implements Variational Bayes.
        :param model: The model
        :type model: pyfilter.timeseries.StateSpaceModel
        :param num_samples: The number of samples
        :type num_samples: int
        :param fullrank: Whether to use full rank approximation
        :type fullrank: bool
        """

        super().__init__(None)
        self._model = model
        self._numsamples = num_samples
        self._fullrank = fullrank

        self._weightmat = torch.ones((self._numsamples, self._model.hidden_ndim))

    def fit(self, y):
        # ===== Get shape of state vectors ====== #
        mean = torch.zeros((self._numsamples, y.shape[0] + 1, self._model.hidden_ndim))

        if not self._fullrank:
            logstd = torch.ones_like(mean)

            eta = dists.Independent(dists.Normal(torch.zeros_like(mean), torch.ones_like(logstd)), 1)

            approx_dist = dists.TransformedDistribution(eta, dists.AffineTransform(mean, logstd.exp()))
        else:
            raise NotImplementedError('Full-rank is currently not implemented!')

        # ===== Start optimization ===== #
        for i in range(1000):
            # ===== Sample ===== #
            samples = approx_dist.sample()
            samples.requires_grad = True

            # ===== Helpers ===== #
            x_t = samples[:, 1:]
            x_tm1 = samples[:, :-1]

            # ===== Loss function ===== #
            logl = (self._model.weight(y, x_t) + self._model.h_weight(x_t, x_tm1)).sum(1)
            logl.backward(self._weightmat)

            # ===== Gradients ===== #
            m_grad = samples.grad.mean(0)

            if not self._fullrank:
                logstd_grad = (samples.grad * samples * logstd.exp()).mean(0) + 1

