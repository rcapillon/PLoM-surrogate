import numpy as np
from scipy.stats import truncnorm, beta, multivariate_normal

from PLoM_surrogate.models import model_sinc


def generator_U(n_samples):
    """

    Parameters
    ----------
    n_samples: number of desired samples

    Returns
    -------
    mat_U: 2xn_samples matrix of realizations of random vector U

    """
    u1 = beta.rvs(2, 5, size=n_samples) + 6
    u0 = truncnorm.rvs(-2, 2, loc=1, scale=1., size=n_samples) + u1 - 6.
    U = np.zeros((2, n_samples))
    U[0, :] = u0
    U[1, :] = u1

    return U


def generator_W(n_samples):
    """

    Parameters
    ----------
    n_samples: number of desired samples

    Returns
    -------
    mat_W: 2xn_samples matrix of realizations of control vector W

    """
    W = np.random.rand(2, n_samples)

    W[0, :] *= 2
    W[0, :] += 1

    W[1, :] *= 2

    return W


def generator_mat_N(nu, m):
    cov = np.eye(nu)
    mat_N = multivariate_normal.rvs(mean=np.zeros((nu, )), cov=cov, size=m).transpose()

    return mat_N
