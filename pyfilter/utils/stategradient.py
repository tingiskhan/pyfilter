from ..timeseries import StateSpaceModel
import autograd as ag


class StateGradient(object):
    def __init__(self, model):
        """
        Implements a way for calculating the gradient and hessian of the underlying states.
        :param model: The model
        :type model: StateSpaceModel
        """

        self._model = model

    @property
    def _ograd(self):
        return ag.elementwise_grad(self._model.weight, argnum=1)

    @property
    def _hgrad(self):
        return ag.elementwise_grad(self._model.h_weight, argnum=0)

    @property
    def _ohess(self):
        return ag.elementwise_grad(self._ograd, argnum=1)

    @property
    def _hhess(self):
        return ag.elementwise_grad(self._hgrad, argnum=0)

    def gradient(self, y, x, oldx):
        """
        Calculates the gradient of both the hidden and observable state.
        :param y: The observation
        :param x: At which point to evaluate the gradient
        :param oldx: The old x
        :return:
        """

        return self._ograd(y, x) + self._hgrad(x, oldx)

    def hess(self, y, x, oldx):
        """
        Calculates the hessian of both the hidden and observable state.
        :param y: The observation
        :param x: At which point to evaluate the gradient
        :param oldx: The old x
        :return:
        """

        return self._ohess(y, x) + self._hhess(x, oldx)

    def o_gradient(self, y, x, oldx):
        """
        Calculates the gradient of the function at x.
        :param y: The observation
        :param x: At which point to evaluate the gradient
        :param oldx: The old x
        :return:
        """

        return self._ograd(y, x)

    def o_hess(self, y, x, oldx):
        """
        Calculates the hessian of the function at x.
        Calculates the gradient of the function at x.
        :param y: The observation
        :param x: At which point to evaluate the gradient
        :param oldx: The old x
        :return:
        """

        return self._ohess(y, x)