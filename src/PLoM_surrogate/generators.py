import numpy as np
from scipy.stats import truncnorm, beta, multivariate_normal

from PLoM_surrogate.models import model_sinc
from PLoM_surrogate.dmaps import compute_L


def generator_U(n_samples):
    """

    Parameters
    ----------
    n_samples: number of desired samples

    Returns
    -------
    mat_U: 2 x n_samples matrix of realizations of random vector U

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
    mat_W: 2 x n_samples matrix of realizations of control vector W

    """
    W = np.random.rand(2, n_samples)

    W[0, :] *= 2
    W[0, :] += 1

    W[1, :] *= 2

    return W


def generator_mat_N(nu, m):
    """

    Parameters
    ----------
    nu: number of dimensions of random multivariate centered normal vector with identity covariance matrix
    m: number of independant samples

    Returns
    -------
    mat_N: nu x m matrix whose columns are independant realizations of multivariate centered normal vectors
    with identity covariance matrix

    """
    cov = np.eye(nu)
    mat_N = multivariate_normal.rvs(mean=np.zeros((nu, )), cov=cov, size=m).transpose()

    return mat_N


def generator_delta_Wiener(nu, N, delta_r):
    """"""
    delta_Wiener = multivariate_normal.rvs(mean=np.zeros((nu * N, )), cov=delta_r * np.eye(nu * N), size=1)
    mat_delta_Wiener = np.reshape(delta_Wiener, shape=(nu, N))

    return mat_delta_Wiener


def generator_ISDE(mat_eta, mat_a, mat_g, delta_r, f_0, M_0, n_MC):
    """
    Generator for additional realizations of a vector random variable from initial sample matrix mat_eta,
    using the reduced diffusion maps basis

    Parameters
    ----------
    mat_eta: nu x N matrix of N realizations of vector random variable concentrated on a manifold
    mat_a: matrix used to project on the diffusion maps basis
    mat_g: matrix used to revert the projection onto the diffusion maps basis
    delta_r: time-step increment in the ISDE generator
    f_0: damping parameter for the ISDE generator
    M_0: number of burned realizations from the ISDE generator to ensure independent realizations as output
    n_MC: number of additional matrices of realizations of mat_eta concentrated on the same manifold

    Returns
    -------

    """
    nu = mat_eta.shape[0]
    N = mat_eta.shape[1]
    m = mat_g.shape[1]

    b = f_0 * delta_r / 4

    mat_N = generator_mat_N(nu, N)

    mat_delta_Wiener_prev = generator_delta_Wiener(nu, N, delta_r)
    mat_Z_proj_prev = np.dot(mat_eta, mat_a)
    mat_Y_proj_prev = np.dot(mat_N, mat_a)

    mat_Z_proj_MC = np.zeros((nu, m * n_MC))

    for i in range(n_MC):
        mat_Z_proj_next = None
        for j in range(M_0):
            mat_Z_proj_prev_half = mat_Z_proj_prev + delta_r * mat_Y_proj_prev / 2
            mat_L_i_half = compute_L(np.dot(mat_Z_proj_prev_half, mat_g.T), mat_eta)
            mat_L_proj_i_half = np.dot(mat_L_i_half, mat_a)
            mat_Y_proj_next = (((1 - b) / (1 + b)) * mat_Y_proj_prev
                               + (delta_r / (1 + b)) * mat_L_proj_i_half
                               + (np.sqrt(f_0) / (1 + b)) * mat_delta_Wiener_prev)
            mat_Z_proj_next = mat_Z_proj_prev_half + delta_r * mat_Y_proj_next / 2

        mat_Z_proj_MC[:, (i * m):((i + 1) * m)] = mat_Z_proj_next

    mat_eta_MC = np.dot(mat_Z_proj_MC, mat_g.T)

    return 0