import numpy as np
from scipy.stats import truncnorm, beta


def generator_U(n_samples):
    """

    Parameters
    ----------
    n_samples: number of desired samples

    Returns
    -------
    mat_U: 2xn_samples matrix of realizations of random vector U

    """

    u0 = truncnorm.rvs(5, 7, loc=2*np.pi, scale=0.5, size=n_samples)
    print(u0.shape)
    u1 = beta.rvs(2, 5, size=n_samples) + 5
    print(u1.shape)
    print(n_samples)
    U = np.zeros((2, n_samples))
    U[0, :] = u0
    U[1, :] = u1

    return U