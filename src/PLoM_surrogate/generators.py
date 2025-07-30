import numpy as np
from scipy.stats import truncnorm, beta

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


def generate_dataset(W, t, n_samples):
    U_samples = generator_U(n_samples)
    rand_Y = np.zeros((n_samples, len(t)))
    for i in range(n_samples):
        U = U_samples[:, i]
        rand_Y[i, :] = model_sinc(W, U, t)

    dataset = np.zeros((3, n_samples * t.size))

    return dataset