from .base import BatchAlgorithm
import torch
from torch import optim
import tqdm
from .varapprox import StateMeanField, BaseApproximation, ParameterMeanField
from ..filters.base import BaseFilter
from ..timeseries import StateSpaceModel
from .utils import stacker
from ..utils import EPS, unflattify


class VariationalBayes(BatchAlgorithm):
    def __init__(self, model, num_samples=4, approx=None, optimizer=optim.Adam, maxiters=30e3, optkwargs=None):
        """
        Implements Variational Bayes.
        :param model: The model
        :type model: StateSpaceModel|pyfilter.timeseries.base.TimeseriesBase
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

        self._s_approx = approx or StateMeanField()
        self._p_approx = ParameterMeanField()

        self._p_mask = None

        self._is_ssm = isinstance(model, StateSpaceModel)

        self._opt_type = optimizer
        self._maxiters = int(maxiters)
        self.optkwargs = optkwargs or dict()

        self._runavg = 0.
        self._decay = 0.975

    @property
    def s_approximation(self):
        """
        Returns the resulting variational approximation of the states.
        :rtype: BaseApproximation
        """
        if not self._is_ssm:
            raise ValueError('There is no state approximation!')

        return self._s_approx

    @property
    def p_approximation(self):
        """
        Returns the resulting variational approximation of the parameters.
        :rtype: BaseApproximation
        """

        return self._p_approx

    def sample_params(self):
        """
        Samples parameters from the variational approximation.
        :return: Self
        :rtype: VariationalBayes
        """

        params = self._p_approx.sample(self._numsamples)
        for p, msk in zip(self._model.theta_dists, self._mask):
            p.detach_()
            p[:] = unflattify(p.bijection(params[:, msk]), p.c_shape)

        self._model.viewify_params((self._numsamples, 1))

        return self

    def _initialize(self, y):
        # ===== Sample model in place for a primitive version of initialization ===== #
        self._model.sample_params(self._numsamples)
        _, self._mask = stacker(self._model.theta_dists)    # NB: We create a mask once

        # ===== Setup the parameter approximation ===== #
        self._p_approx = self._p_approx.initialize(self._model.theta_dists)

        params = self._p_approx.get_parameters()

        # ===== Initialize the state approximation ===== #
        if self._is_ssm:
            self._s_approx.initialize(y, self._model.hidden_ndim)
            params += self._s_approx.get_parameters()

        return params

    def loss(self, y):
        """
        The loss function, i.e. ELBO.
        :rtype: torch.Tensor
        """
        # ===== Sample parameters ===== #
        self.sample_params()
        entropy = self._p_approx.entropy()

        if self._is_ssm:
            # ===== Sample states ===== #
            transformed = self._s_approx.sample(self._numsamples)

            # ===== Helpers ===== #
            x_t = transformed[:, 1:]
            x_tm1 = transformed[:, :-1]

            if self._model.hidden_ndim < 2:
                x_t.squeeze_(-1)
                x_tm1.squeeze_(-1)

            logl = self._model.weight(y, x_t) + self._model.h_weight(x_t, x_tm1)
            entropy += self._s_approx.entropy()

        else:
            logl = self._model.weight(y[1:], y[:-1])

        return -(logl.sum(1).mean(0) + torch.mean(self._model.p_prior(transformed=True), dtype=logl.dtype) + entropy)

    def _fit(self, y):
        optparams = self._initialize(y)

        elbo_old = -torch.tensor(float('inf'))
        elbo = -elbo_old

        optimizer = self._opt_type(optparams, **self.optkwargs)

        it = 0
        bar = tqdm.tqdm(total=self._maxiters)
        while (elbo - elbo_old).abs() > EPS and it < self._maxiters:
            elbo_old = elbo

            # ===== Perform optimization ===== #
            optimizer.zero_grad()
            elbo = self.loss(y)
            elbo.backward()
            optimizer.step()

            bar.update(1)

            if it > 0:
                self._runavg = self._runavg * self._decay - elbo * (1 - self._decay)
            else:
                self._runavg = -elbo

            bar.set_description('{:s} - Avg. ELBO: {:.2f}'.format(str(self), self._runavg))
            it += 1

        return self


class VariationalSMC(VariationalBayes):
    def __init__(self, filter_, maxiters=100, **kwargs):
        """
        Implementation of variational smc
        :param filter_: The filter to use
        :type filter_: BaseFilter
        """
        raise NotImplementedError('Currently does not work')

        super().__init__(model=filter_.ssm, maxiters=maxiters, **kwargs)
        self._filter = filter_.set_nparallel(self._numsamples)

    def _initialize(self, y):
        self.filter._rsample = True

        # ===== Sample model in place for a primitive version of initialization ===== #
        self._model.sample_params(self._numsamples)

        # ===== Setup the parameter approximation ===== #
        self._p_approx = self._p_approx.initialize(self._model.theta_dists)

        return self._p_approx.get_parameters()

    def loss(self, y):
        self.filter.reset()

        # ===== Sample parameters ===== #
        self.sample_params()

        # ===== Loss function ===== #
        self.filter.initialize().longfilter(y, bar=False)
        ll = self.filter.loglikelihood.mean()

        return -(ll + self._model.p_prior(transformed=True).mean() + self._p_approx.entropy())
