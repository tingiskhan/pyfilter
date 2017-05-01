import numpy as np
import math


def searchsorted2d(a, b):
    """
    Searches a sorted 2D array along the second axis. Basically performs a vectorized digitize.
    :param a:
    :param b:
    :return:
    """
    m, n = a.shape
    max_num = np.maximum(a.max(), b.max()) + 1
    r = max_num * np.arange(a.shape[0])[:, None]
    p = np.searchsorted((a + r).ravel(), (b + r).ravel()).reshape(m, -1)

    return p - n * np.arange(m)[:, None]


def choose(array, indices):
    """
    Function for choosing on either columns or index.
    :param array:
    :param indices:
    :return:
    """

    if isinstance(array, list):
        out = list()
        for it in array:
            out.append(choose(it, indices))

        return out

    return array[indices]

    # if array.ndim > indices.ndim:
    #     row_mask = np.arange(array.shape[1])[:, None]
    #     return array[:, row_mask, indices]
    #
    # row_mask = np.arange(array.shape[0])[:, None]
    # return array[row_mask, indices]


def solve_quadratic(a, b, c):
    """
    Solves a quadratic function f(x) = ax^2 + bx + c
    :param a:
    :param b:
    :param c:
    :return:
    """
    return (-b + math.sqrt(b ** 2 - 4 * a * c)) / 2 / a


def solve_sum(s):
    """
    Backs out the number of integers required to sum up to `s`.
    :param s:
    :return:
    """
    return solve_quadratic(1, 1, -2 * s)


def makediag3d(a):
    """
    Takes a 2D array `a` as inputs and constructs a diagonal 3D array from the rows of the 2D array.
    :param a:
    :return:
    """
    size = a.shape[-1]
    x = np.zeros((a.shape[0], a.shape[1], a.shape[1]))
    mask = np.eye(size) == np.ones((size, size))

    x[:, mask] = a

    return x


def getdiag3d(a):
    """
    Flattens a diagonal 3D array to 2D.
    :param a:
    :return:
    """
    size = a.shape[-1]
    mask = np.eye(size) == np.ones((size, size))

    x = a[:, mask]

    return x


def makeuppdiag3d(a):
    """
    Constructs a 3D upper diagonal matrix given a 2D array where each row corresponds to the upper diagonal matrix.
    :param a:
    :return:
    """
    size = solve_sum(a.shape[-1])

    x = np.zeros((a.shape[0], size, size))
    mask = np.arange(size)[:, None] <= np.arange(size)

    x[:, mask] = a

    return x


def makediag4d(a):
    size = a.shape[-1]
    x = np.zeros((a.shape[0], a.shape[1], a.shape[2], a.shape[2]))
    mask = np.eye(size) == np.ones((size, size))

    x[:, :, mask] = a

    return x


def makeuppdiag4d(a):
    size = solve_sum(a.shape[2])

    x = np.zeros((a.shape[0], a.shape[1], size, size))
    mask = np.arange(size)[:, None] <= np.arange(size)

    x[:, :, mask] = a

    return x


def expand_dimensions(a, b):
    """
    Expands the dimension of `a` to be that of `b`.
    :param a:
    :param b:
    :return:
    """
    try:
        if a.shape == b.shape:
            return a
        elif (a.ndim == b.ndim) and (a.shape[0] == b.shape[0]):
            return a
        elif a.ndim > 1:
            return a[:, :, None]

        return a[None, :, None]

    except AttributeError:
        return a


def standardize(data):
    """
    Standardizes data.
    :type data: pandas.DataFrame
    :rtype: pandas.DataFrame
    """

    return data.subtract(data.mean()).div(data.std())


def summation_axis(array):
    tmp = tuple([ax for ax in range(array.ndim) if ax > 0])
    return tmp if len(tmp) > 0 else 0


