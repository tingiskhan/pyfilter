import numpy as np
from collections import Iterable
from .normalization import normalize


def get_ess(w):
    """
    Calculates the ESS from an array of log weights.
    :param w: The log weights
    :type w: np.ndarray
    :return: The effective sample size
    :rtype: float
    """

    normalized = normalize(w)

    return np.sum(normalized, axis=-1) ** 2 / np.sum(normalized ** 2, axis=-1)


def searchsorted2d(a, b):
    """
    Searches a sorted 2D array along the second axis. Basically performs a vectorized digitize. Solution provided here:
    https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy.
    :param a: The array to take the elements from
    :type a: np.ndarray
    :param b: The indices of the elements in `a`.
    :type b: np.ndarray
    :return: An array of indices
    :rtype: np.ndarray
    """
    m, n = a.shape
    max_num = np.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * np.arange(a.shape[0])[:, None]
    p = np.searchsorted((a + r).ravel(), (b + r).ravel()).reshape(m, -1)
    return p - n * np.arange(m)[:, None]


def choose(array, indices):
    """
    Function for choosing on either columns or index.
    :param array: The array to choose on
    :type array: np.ndarray
    :param indices: The indices to choose from `array`
    :type indices: np.ndarray
    :return: Returns the chosen elements from `array`
    :rtype: np.ndarray
    """

    if isinstance(array, list):
        out = list()
        for it in array:
            out.append(choose(it, indices))

        return out

    if indices.ndim < 2:
        shapematch = np.cumsum([s == indices.shape[0] for s in array.shape])
        return np.take(array, indices, axis=shapematch.tolist().index(1))

    return array[..., np.arange(array.shape[-2])[:, None], indices]


def loglikelihood(w):
    """
    Calculates the estimated loglikehood given weights.
    :param w: The log weights, corresponding to likelihood
    :type w: np.ndarray
    :return: The log-likelihood
    :rtype: np.ndarray
    """

    maxw = np.max(w, axis=-1)

    return maxw + np.log(np.exp(w.T - maxw).T.mean(axis=-1))


def dot(a, b):
    """
    Helper function for calculating the dot product between a matrix and a vector.
    :param a: The A array
    :type a: np.ndarray
    :param b: The B vector
    :type b: np.ndarray
    :return: Returns the matrix product
    :rtype: np.ndarray
    """

    return np.einsum('ij...,j...->i...', a, b)


def outer(a, b):
    """
    Calculates the outer product B * X * B^t.
    :param a: The B-matrix
    :type a: np.ndarray
    :param b: The X-matrix
    :type b: np.ndarray
    :return: The outer product
    :rtype: np.ndarray
    """

    xbt = np.einsum('ij...,kj...->ik...', b, a)
    bxbt = np.einsum('ij...,jk...->ik...', a, xbt)

    return bxbt


def square(a, b):
    """
    Calculates the square product x^t * A * x.
    :param a: The vector x
    :type a: np.ndarray
    :param b: The matrix A
    :type b: np.ndarray
    :return: The squared matrix
    :rtype: np.ndarray
    """

    return np.einsum('i...,i...->...', a, dot(b, a))


def outerv(a, b):
    """
    Performs the outer multiplication of the expression a * b^t.
    :param a: The vector a
    :type a: np.ndarray
    :param b: The vector b
    :type b: np.ndarray
    :return: The outer product
    :rtype: np.ndarray
    """

    return np.einsum('i...,j...->ij...', a, b)


def expanddims(array, ndims):
    """
    Expand dimensions to the right to target the dimensions of ndims.
    :param array: The current array
    :type array: np.ndarray
    :param ndims: The target number of dimensions.
    :type ndims: int
    :return: Expanded array
    :rtype: np.ndarray
    """

    if array.ndim == ndims:
        return array

    copy = np.expand_dims(array, -1)

    return expanddims(copy, ndims)


def mdot(a, b):
    """
    Performs the matrix dot product `a * b`.
    :param a: The a matrix
    :type a: np.ndarray
    :param b: The b matrix
    :type b: np.ndarray
    :return: The matrix dot product
    :rtype: np.ndarray
    """

    return np.einsum('ij...,jk...->ik...', a, b)


def outerm(a, b):
    """
    Calculates the outer matrix product, ie A * B^t
    :param a: The array a
    :type a: np.ndarray
    :param b: The array b
    :type b: np.ndarray
    :return: The outer matrix product
    :rtype: np.ndarray
    """

    return np.einsum('ij...,kj...->ik...', a, b)


def customcholesky(a):
    """
    Performs a custom Cholesky factorization of the array a.
    :param a: The array a
    :type a: np.ndarray
    :return: Cholesky decomposition of `a`
    :rtype: np.ndarray
    """

    firstaxes = (*range(2, 2 + len(a.shape[2:])), 0, 1)
    secondaxes = (-2, -1, *range(len(a.shape[2:])))

    return np.linalg.cholesky(a.transpose(firstaxes)).transpose(secondaxes)


def resizer(tup):
    """
    Recasts all the non-array elements of an array to arrays of the same size as the array elements. If you for example
    pass an array `[[0, a], [a, 0]]`, where `a` is a numpy.ndarray, then the elements `0` will be recast into arrays
    of the same size as `a`. Implementation of
    https://stackoverflow.com/questions/47640385/create-array-from-mixed-floats-and-arrays-python
    :param tup: The array to recast.
    :type tup: Iterable
    :return: Resized array
    :rtype: np.ndarray
    """
    # TODO: Speed up
    if isinstance(tup, (int, float, np.ndarray)):
        return tup

    asarray = np.array(tup, dtype=object)
    flat = flatten(tup)
    shape = next((e.shape for e in flat if isinstance(e, np.ndarray)), False)

    if not shape or asarray.shape[-len(shape):] == shape:
        return np.array(tup)

    out = np.empty((len(flat), *shape))
    for i, e in enumerate(flat):
        out[i] = e

    try:
        return out.reshape((*asarray.shape, *shape))
    except ValueError as e:
        raise ValueError('Most likely errors in the dimension!') from e


def flatten(iterable):
    """
    Flattens an array comprised of an arbitrary number of lists. Solution found at:
        https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    :param iterable: The iterable you wish to flatten.
    :type iterable: collections.Iterable
    :return:
    """
    out = list()
    for el in iterable:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes, np.ndarray)):
            out.extend(flatten(el))
        else:
            out.append(el)

    return out

