import unittest
from pyfilter.timeseries import AffineProcess, AffineObservations, StateSpaceModel
from torch.distributions import Normal, MultivariateNormal, Independent
from pyfilter.unscentedtransform import UnscentedTransform
import torch
from pyfilter.filters import SISR, UKF
from pyfilter.module import Module, TensorContainer
from pyfilter.utils import concater
from pyfilter.inference.utils import stacker
from pyfilter.timeseries import Parameter


class Help2(Module):
    def __init__(self, a):
        self.a = a


class Help(Module):
    def __init__(self, *params):
        self._params = TensorContainer(*params)
        self._views = TensorContainer(p.view(-1) for p in params)
        self._mod = Help2(self._params[0] + 1)


def f(x, alpha, sigma):
    return alpha * x


def g(x, alpha, sigma):
    return sigma


def f0(alpha, sigma):
    return 0


def g0(alpha, sigma):
    return sigma


def fo(x, alpha, sigma):
    return alpha * x


def go(x, alpha, sigma):
    return sigma


def foo(x1, x2, alpha, sigma):
    return alpha * x1 + x2


def goo(x1, x2, alpha, sigma):
    return sigma


def fmvn(x, a, sigma):
    return concater(x[..., 0], x[..., 1])


def f0mvn(a, sigma):
    return torch.zeros(2)


def fomvn(x, sigma):
    return x[..., 0] + x[..., 1] / 2


def gomvn(x, sigma):
    return sigma


class Tests(unittest.TestCase):
    def test_UnscentedTransform1D(self):
        # ===== 1D model ===== #
        norm = Normal(0., 1.)
        linear = AffineProcess((f, g), (1., 1.), norm, norm)
        linearobs = AffineObservations((fo, go), (1., 1.), norm)
        model = StateSpaceModel(linear, linearobs)

        # ===== Perform unscented transform ===== #
        x = model.hidden.i_sample(shape=3000)

        ut = UnscentedTransform(model).initialize(x).construct(0.)

        assert isinstance(ut.x_dist, Normal)

    def test_UnscentedTransform2D(self):
        # ===== 2D model ===== #
        mat = torch.eye(2)
        scale = torch.diag(mat)

        norm = Normal(0., 1.)
        mvn = MultivariateNormal(torch.zeros(2), torch.eye(2))
        mvnlinear = AffineProcess((fmvn, g), (mat, scale), mvn, mvn)
        mvnoblinear = AffineObservations((fomvn, gomvn), (1.,), norm)

        mvnmodel = StateSpaceModel(mvnlinear, mvnoblinear)

        # ===== Perform unscented transform ===== #
        x = mvnmodel.hidden.i_sample(shape=3000)

        ut = UnscentedTransform(mvnmodel).initialize(x).construct(0.)

        assert isinstance(ut.x_dist, MultivariateNormal) and isinstance(ut.y_dist, Normal)
        assert isinstance(ut.x_dist_indep, Independent)

    def test_HelperMixin(self):
        obj = Help(torch.empty(3000).normal_())

        # ===== Verify that we don't break views when changing device ===== #
        obj.to_('cpu:0')

        temp = obj._params[0]
        temp += 1

        for p, v in zip(obj._params, obj._views):
            assert (p == v).all() and v._base is p

        # ===== Check state dict ===== #
        sd = obj.state_dict()

        newobj = Help(torch.empty(1))
        newobj.load_state_dict(sd)

        assert all((p1 == p2).all() for p1, p2 in zip(newobj._params, obj._params))
        assert all((p1 == p2).all() for p1, p2 in zip(newobj._views, newobj._params))
        assert all(p1._base is p2 for p1, p2 in zip(newobj._views, newobj._params))

    def test_StateDict(self):
        # ===== Define model ===== #
        norm = Normal(0., 1.)
        linear = AffineProcess((f, g), (1., 1.), norm, norm)
        linearobs = AffineObservations((fo, go), (1., 1.), norm)
        model = StateSpaceModel(linear, linearobs)

        # ===== Define filter ===== #
        filt = SISR(model, 100).initialize()

        # ===== Get statedict ===== #
        sd = filt.state_dict()

        # ===== Verify that we don't save multiple instances ===== #
        assert '_model' in sd and '_model' not in sd['_proposal']

        newfilt = SISR(model, 1000).load_state_dict(sd)
        assert newfilt._w_old is not None and newfilt.ssm is newfilt._proposal._model

        # ===== Test same with UKF and verify that we save UT ===== #
        ukf = UKF(model).initialize()
        sd = ukf.state_dict()

        assert '_model' in sd and '_model' not in sd['_ut']

    def test_Stacker(self):
        # ===== Define a mix of parameters ====== #
        zerod = Parameter(Normal(0., 1.)).sample_((1000,))
        oned_luring = Parameter(Normal(torch.tensor([0.]), torch.tensor([1.]))).sample_(zerod.shape)
        oned = Parameter(MultivariateNormal(torch.zeros(2), torch.eye(2))).sample_(zerod.shape)

        mu = torch.zeros((3, 3))
        norm = Independent(Normal(mu, torch.ones_like(mu)), 2)
        twod = Parameter(norm).sample_(zerod.shape)

        # ===== Stack ===== #
        params = (zerod, oned, oned_luring, twod)
        stacked = stacker(params, lambda u: u.t_values, dim=1)

        # ===== Verify it's recreated correctly ====== #
        for p, m, ps in zip(params, stacked.mask, stacked.prev_shape):
            v = stacked.concated[..., m]

            if len(p.c_shape) != 0:
                v = v.reshape(*v.shape[:-1], *ps)

            assert (p.t_values == v).all()
