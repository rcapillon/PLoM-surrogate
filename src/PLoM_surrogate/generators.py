import numpy as np
from scipy.stats import truncnorm, beta, multivariate_normal, Covariance
from tqdm import tqdm

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


def generator_Delta_Wiener(nu, N, delta_r):
    """

    Parameters
    ----------
    nu: number of rows in the increment matrix for the Wiener process
    N: number of columns in the increment matrix for the Wiener process
    delta_r: time-step increment in the ISDE generator

    Returns
    -------
    mat_Delta_Wiener: nu x N matrix of increment for the Wiener process

    """
    cov = Covariance.from_diagonal(np.array([delta_r] * (nu * N)))
    Delta_Wiener = multivariate_normal.rvs(mean=np.zeros((nu * N, )), cov=cov, size=1)
    mat_Delta_Wiener = np.reshape(Delta_Wiener, shape=(nu, N))

    return mat_Delta_Wiener


def generator_ISDE(dataset, mat_a, mat_g, delta_r, f_0, M_0, n_MC, progress_bar=True):
    """
    Generator for additional realizations of a vector random variable from initial sample matrix mat_eta,
    using the reduced diffusion maps basis

    Parameters
    ----------
    dataset: dataset containing the training dataset with PCA applied on outputs and then globally
    mat_a: matrix used to project on the diffusion maps basis
    mat_g: matrix used to revert the projection onto the diffusion maps basis
    delta_r: time-step increment in the ISDE generator
    f_0: damping parameter for the ISDE generator
    M_0: number of burned realizations from the ISDE generator to ensure independent realizations as output
    n_MC: number of additional matrices of realizations concentrated on the same manifold as the training dataset

    Returns
    -------
    data_MCMC: additional independent realizations concentrated on the same manifold as the training dataset

    """
    nu = dataset.H_data.shape[0]
    N = dataset.H_data.shape[1]

    b = f_0 * delta_r / 4

    mat_N = generator_mat_N(nu, N)
    mat_Delta_Wiener_prev = generator_Delta_Wiener(nu, N, delta_r)
    mat_Delta_Wiener_proj_prev = np.dot(mat_Delta_Wiener_prev, mat_a)
    mat_Z_proj_prev = np.dot(dataset.H_data, mat_a)
    mat_Y_proj_prev = np.dot(mat_N, mat_a)

    mat_eta_MC = np.zeros((nu, N * n_MC))

    for i in tqdm(range(n_MC), disable=not progress_bar):
        mat_Z_proj_next = None
        for j in range(M_0):
            mat_Z_proj_prev_half = mat_Z_proj_prev + delta_r * mat_Y_proj_prev / 2
            mat_L_i_half = compute_L(np.dot(mat_Z_proj_prev_half, mat_g.T), dataset.H_data)
            mat_L_proj_i_half = np.dot(mat_L_i_half, mat_a)
            mat_Y_proj_next = (((1 - b) / (1 + b)) * mat_Y_proj_prev
                               + (delta_r / (1 + b)) * mat_L_proj_i_half
                               + (np.sqrt(f_0) / (1 + b)) * mat_Delta_Wiener_proj_prev)
            mat_Z_proj_next = mat_Z_proj_prev_half + delta_r * mat_Y_proj_next / 2

            mat_Z_proj_prev = mat_Z_proj_next
            mat_Y_proj_prev = mat_Y_proj_next
            mat_Delta_Wiener_prev = generator_Delta_Wiener(nu, N, delta_r)
            mat_Delta_Wiener_proj_prev = np.dot(mat_Delta_Wiener_prev, mat_a)

        mat_eta_i = np.dot(mat_Z_proj_next, mat_g.T)
        mat_eta_MC[:, (i * N):((i + 1) * N)] = mat_eta_i

    X_MCMC = dataset.recover_X(mat_eta_MC)
    data_MCMC = dataset.recover_data(X_MCMC)

    return data_MCMC