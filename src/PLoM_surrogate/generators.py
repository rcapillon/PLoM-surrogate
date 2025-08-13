import numpy as np
from scipy.stats import multivariate_normal, Covariance
from multiprocessing import Pool
from tqdm import tqdm

from PLoM_surrogate.dmaps import construct_dmaps_basis, build_mat_a


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
    progress_bar: if True, displays a progress bar for the generation of additional realizations

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


class Generator:
    def __init__(self, dataset, n_cpu,
                 Fac=20, delta_r=None, f_0=1.5, M_0=100, eps=3., kappa=1, m=30):
        """"""
        self.dataset = dataset
        self.n_cpu = n_cpu
        self.Fac = 20
        if delta_r is None:
            s_nu = np.power(4 / (self.dataset.H_data.shape[1] * (2 + dataset.H_data.shape[0])),
                            1 / (self.dataset.H_data.shape[0] + 4))
            s_hat_nu = s_nu / (np.sqrt(s_nu ** 2 + ((self.dataset.H_data.shape[1] - 1) / self.dataset.H_data.shape[1])))
            self.delta_r = 2 * np.pi * s_hat_nu / Fac
        else:
            self.delta_r = delta_r
        self.f_0 = f_0
        self.M_0 = M_0
        self.eps = eps
        self.kappa = kappa
        self.m = m

        self.mat_g = None
        self.mat_a = None
        
    def construct_dmaps_basis(self, plot_eigvals_name=None):
        """"""
        self.mat_g = construct_dmaps_basis(self.dataset.H_data, self.eps, self.m, self.kappa,
                                           plot_eigvals_name=plot_eigvals_name)
        self.mat_a = build_mat_a(self.mat_g)

    def generate_realizations(self, n_MC):
        """"""
        pool = Pool(processes=self.n_cpu)
        total_data_MCMC = np.empty((self.dataset.dim, self.dataset.n_t, 0))
        progress_bar = True
        inputs = ([(self.dataset, self.mat_a, self.mat_g, self.delta_r, self.f_0, self.M_0, n_MC, progress_bar)]
                  * self.n_cpu)

        for data_MCMC in pool.starmap(generator_ISDE, inputs):
            total_data_MCMC = np.concatenate((total_data_MCMC, data_MCMC), axis=-1)

        return total_data_MCMC