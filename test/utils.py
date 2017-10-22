import unittest
import pyfilter.utils.utils as helps
import numpy as np


class Tests(unittest.TestCase):
    def test_OuterProduct(self):
        a = np.random.normal(size=(3, 3))
        b = np.random.normal(size=(3, 3))

        true = a.dot(b.dot(a.T))
        est = helps.outer(a, b)

        assert np.allclose(true, est)

    def test_Dot(self):
        a = np.random.normal(size=(2, 2))
        b = np.random.normal(size=2)
        
        trueval = a.dot(b)

        est = helps.dot(a, b)

        assert np.allclose(trueval, est)