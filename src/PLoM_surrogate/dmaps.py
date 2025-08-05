import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(vec1, vec2, eps):
    """
    Computes gaussian kernel for two vectors with a smoothing parameter

    Parameters
    ----------
    vec1: first vector used in gaussian kernel
    vec2: second vector used in gaussian kernel
    eps: smoothing parameter (must be positive)

    Returns
    -------
    k: scalar kernel value for the given arguments

    """
    k = np.exp((-1. / (4 * eps)) * np.linalg.norm(vec1 - vec2) ** 2)

    return k


def build_mat_K(mat_eta, eps):
    """

    Parameters
    ----------
    mat_eta: matrix of random realizations of a vector random variable
    eps: smoothing parameter (must be positive) used for the kernel

    Returns
    -------
    mat_K: matrix of kernel values for all realizations of a vector random variable contained in columns of mat_eta

    """
    N = mat_eta.shape[1]
    mat_K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mat_K[i, j] = gaussian_kernel(mat_eta[:, i], mat_eta[:, j], eps)

    return mat_K


def build_mat_b(mat_K):
    """

    Parameters
    ----------
    mat_K: matrix containing the kernel values for all realizations of a vector random variable

    Returns
    -------
    mat_b: normalization matrix for input matrix mat_K

    """
    N = mat_K.shape[0]
    mat_b = np.zeros((N, N))
    for i in range(N):
        mat_b[i, i] = np.sum(mat_K[i, :])

    return mat_b


def build_mat_Ps(mat_b, mat_K):
    """

    Parameters
    ----------
    mat_b: normalization matrix for mat_K
    mat_K: matrix containing the kernel values for all realizations of a vector random variable

    Returns
    -------
    mat_Ps: symmetrized transition matrix for the underlying MCMC generator

    """
    inv_sqrt_b = np.linalg.inv(np.sqrt(mat_b))
    mat_Ps = np.dot(inv_sqrt_b, np.dot(mat_K, inv_sqrt_b))
    mat_Ps = 0.5 * (mat_Ps + mat_Ps.T)

    return mat_Ps


def construct_dmaps_basis(mat_eta, eps, m, kappa, plot_eigvals=False):
    """

    Parameters
    ----------
    mat_eta: matrix of random realizations of a vector random variable
    eps: smoothing parameter (must be positive) used for the kernel
    m: number of retained diffusion maps basis vectors
    kappa: number of steps used to calculate diffusion distance

    Returns
    -------
    mat_g: diffusion maps basis constructed with realizations of vector random variable in columns of matrix mat_eta

    """
    mat_K = build_mat_K(mat_eta, eps)
    mat_b = build_mat_b(mat_K)
    mat_Ps = build_mat_Ps(mat_b, mat_K)

    eigvals, eigvects = np.linalg.eig(mat_Ps)
    inds_sort_eigvals = np.flip(np.argsort(eigvals))
    sorted_eigvals = np.real(eigvals[inds_sort_eigvals])
    sorted_eigvects = np.real(eigvects[:, inds_sort_eigvals])

    if plot_eigvals:
        _, ax = plt.subplots()
        ax.semilogy(sorted_eigvals, '-b')
        ax.set_title('eigenvalues of [Ps]')
        ax.set_xlabel('index')
        ax.set_ylabel('eigenvalues')
        plt.grid()
        plt.savefig('./test_dmaps_eigenvalues.png')

    vec_lambda = sorted_eigvals[:m]
    mat_phi = sorted_eigvects[:, :m]

    inv_sqrt_b = np.linalg.inv(np.sqrt(mat_b))
    mat_psi = np.dot(inv_sqrt_b, mat_phi)

    mat_g = np.zeros((mat_psi.shape[0], m))
    for i in range(m):
        mat_g[:, i] = (vec_lambda[i] ** kappa) * mat_psi[:, i]

    return mat_g


def build_mat_a(mat_g):
    """"""
    inv_gTg = np.linalg.inv(np.dot(mat_g.T, mat_g))
    mat_a = np.dot(mat_g, inv_gTg)

    return mat_a


def project_on_dmaps(mat, mat_a):
    """

    Parameters
    ----------
    mat: matrix to be projected on the diffusion maps reduced basis
    mat_a: matrix used to project on the diffusion maps basis

    Returns
    -------

    """
    projected_mat = np.dot(mat, mat_a)

    return projected_mat


def reverse_project_from_dmaps(projected_mat, mat_g):
    """

    Parameters
    ----------
    projected_mat: matrix for which the projection onto the reduced diffusion maps basis is to be reverted
    mat_g: matrix used to revert the projection onto the diffusion maps basis

    Returns
    -------

    """
    mat = np.dot(projected_mat, mat_g.T)

    return mat


def compute_q(u, mat_eta):
    """

    Parameters
    ----------
    u: vector-valued realization of stochastic process U at current step
    mat_eta: matrix whose columns are realizations of a vector random variable

    Returns
    -------
    q: value used to calculate the potential used in the ISDE generator

    """
    nu = mat_eta.shape[0]
    N = mat_eta.shape[1]

    s_nu = np.power(4 / (N * (2 + nu)), 1 / (nu + 4))
    s_hat_nu = s_nu / (np.sqrt(s_nu ** 2 + ((N - 1) / N)))

    q = 0.
    for i in range(N):
        q += np.exp(-np.linalg.norm(s_hat_nu * mat_eta[:, i] / s_nu - u) ** 2 / (2 * s_hat_nu ** 2))
    q /= N

    return q


def compute_grad_q(u, mat_eta):
    """

    Parameters
    ----------
    u: vector-valued realization of stochastic process U at current step
    mat_eta: matrix whose columns are realizations of a vector random variable

    Returns
    -------
    grad_q: gradient of the value used to calculate the potential used in the ISDE generator

    """
    nu = mat_eta.shape[0]
    N = mat_eta.shape[1]

    s_nu = np.power(4 / (N * (2 + nu)), 1 / (nu + 4))
    s_hat_nu = s_nu / (np.sqrt(s_nu ** 2 + ((N - 1) / N)))

    grad_q = np.zeros((nu, ))
    for i in range(N):
        term_1 = (s_hat_nu * mat_eta[:, i] / s_nu) - u
        term_2 = np.exp(-np.linalg.norm(term_1) ** 2 / (2 * s_hat_nu ** 2))
        grad_q += term_1 * term_2
    grad_q /= N * s_hat_nu ** 2

    return grad_q


def compute_L(mat_u, mat_eta):
    """

    Parameters
    ----------
    mat_u: matrix whose columns are vector-valued realizations of stochastic process U at current step
    mat_eta: matrix whose columns are realizations of a vector random variable

    Returns
    -------
    mat_L: value of matrix L used in the ISDE generator

    """
    nu = mat_eta.shape[0]
    N = mat_eta.shape[1]

    mat_L = np.zeros((nu, N))
    for i in range(N):
        u = mat_u[:, i]
        q = compute_q(u, mat_eta)
        grad_q = compute_grad_q(u, mat_eta)
        mat_L[:, i] = grad_q / q

    return mat_L