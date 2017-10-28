import autograd.numpy as np
from .normalization import normalize


def get_ess(w):
    """
    Calculates the ESS from an array of log weights.
    :param w: The log weights
    :type w: np.ndarray
    :return:
    """

    normalized = normalize(w)

    return np.sum(normalized, axis=-1) ** 2 / np.sum(normalized ** 2, axis=-1)


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

    if indices.ndim < 2:
        shapematch = np.cumsum([s == indices.shape[0] for s in array.shape])
        return np.take(array, indices, axis=shapematch.tolist().index(1))

    return array[..., np.arange(array.shape[-2])[:, None], indices]


def loglikelihood(w):
    """
    Calculates the estimated loglikehood given weights.
    :param w: The log weights, corresponding to likelihood
    :type w: np.ndarray
    :return: 
    """

    maxw = np.max(w, axis=-1)

    return maxw + np.log(np.exp(w.T - maxw).T.mean(axis=-1))


def dot(a, b):
    """
    Helper function for calculating the dot product between two matrices.
    :param a: The A array
    :type a: np.ndarray
    :param b: The B array
    :type b: np.ndarray
    :return: 
    """

    return np.einsum('ij...,j...->i...', a, b)


def outer(a, b):
    """
    Calculates the outer product B * X * B^t.
    :param a: The B-matrix
    :param b: The X-matrix
    :return: 
    """

    xbt = np.einsum('ij...,kj...->ik...', b, a)
    bxbt = np.einsum('ij...,jk...->ik...', a, xbt)

    return bxbt


def square(a, b):
    """
    Calculates the square product x^t * A * x.
    :param a: The vector x
    :param b: The matrix A 
    :return: 
    """

    return np.einsum('i...,i...->...', a, dot(b, a))


def outerv(a, b):
    """
    Performs the outer multiplication of the expression a * b^t.
    :param a: The vector a
    :type a: np.ndarray
    :param b: The vector b
    :type b: np.ndarray
    :return:
    """

    return np.einsum('i...,j...->ij...', a, b)


def expanddims(array, ndims):
    """
    Expand dimensions to the right to target the dimensions of ndims.
    :param array: The current array
    :param ndims: The target number of dimensions.
    :return:
    """

    if array.ndim == ndims:
        return array

    copy = np.expand_dims(array, -1)

    return expanddims(copy, ndims)


def mdot(a, b):
    """
    Performs the matrix dot product `a * b`.
    :param a: The a matrix
    :param b: The b matrix
    :return:
    """

    return np.einsum('ij...,jk...->ik...', a, b)


def customcholesky(a):
    """
    Performs a custom Cholesky factorization of the array a.
    :param a: The array a
    :return:
    """

    firstaxes = (*range(2, 2 + len(a.shape[2:])), 0, 1)
    secondaxes = (-2, -1, *range(len(a.shape[2:])))

    return np.linalg.cholesky(a.transpose(firstaxes)).transpose(secondaxes)