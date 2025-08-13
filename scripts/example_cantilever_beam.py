import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from typing import Union
import numpy as np
from scipy.stats import gamma, uniform
import matplotlib.pyplot as plt
import time

from PLoM_surrogate.models import Surrogate
from PLoM_surrogate.generators import Generator
from PLoM_surrogate.data import Dataset


def model_cantilever_beam(W: np.ndarray, U: np.ndarray,
                          x: Union[list, np.ndarray], t: Union[list, np.ndarray],
                          Fmax: float):
    """
    Quasi-static deflection of a cantilever beam of length 1m, subjected to a downwards concentrated load
    at its free end.
    Source: https://home.engineering.iastate.edu/~shermanp/STAT447/STAT%20Articles/Beam_Deflection_Formulae.pdf

    Parameters
    ----------
    W: Control parameter, corresponding to application point of the concentrated load
    U: Uncertain parameters, corresponding to the Young's modulus and the second moment of area
    x: Abscissa along the beam, between 0 and 1
    t: List or numpy vector of Nt pseudo-time values between 0 and 1 for which the output is calculated
    Fmax: Constant parameter, corresponding to the maximal value for the load.
        The load evolves linearly with t, reaching Fmax at t=1

    Returns
    -------
    y: Nx x Nt array of deflections for the beam

    """
    if W.ndim != 1 or W.size != 1:
        raise ValueError('W must be a numpy vector with 1 components.')
    if U.ndim != 1 or U.size != 2:
        raise ValueError('U must be a numpy vector with 2 components.')
    if isinstance(x, list):
        arr_x = np.array(x)
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        arr_x = x
    else:
        raise TypeError('Argument x must be a list or a numpy vector.')
    if isinstance(t, list):
        arr_t = np.array(t)
    elif isinstance(t, np.ndarray) and t.ndim == 1:
        arr_t = t
    else:
        raise TypeError('Argument t must be a list or a numpy vector.')
    if Fmax <= 0:
        raise ValueError('Argument Fmax must be strictly positive.')

    Y = np.zeros((x.size, t.size))
    for j in range(t.size):
        F = arr_t[j] * Fmax
        for i in range(x.size):
            x_i = arr_x[i]
            if x_i <= W[0]:
                Y[i, j] = -F * (x_i ** 2) * (3 * W[0] - x_i) / (6 * U[0] * U[1])
            else:
                Y[i, j] = -F * (W[0] ** 2) * (3 * x_i - W[0]) / (6 * U[0] * U[1])

    return Y


def generator_E_cantilever(n_samples):
    """"""
    mean_E = 2.1e11
    dispersion_coeff = 0.1
    std = mean_E * dispersion_coeff
    a = 1 / dispersion_coeff ** 2
    b = (std ** 2) / mean_E
    E_samples = gamma.rvs(a, scale=b, size=n_samples)
    E = np.zeros((1, n_samples))
    E[0, :] = E_samples

    return E


def generator_I_cantilever(n_samples):
    """"""
    D = uniform.rvs(loc=0.8, scale=1.2, size=n_samples)
    I_samples = np.pi * np.power(D, 4) / 64.
    I = np.zeros((1, n_samples))
    I[0, :] = I_samples

    return I


def generate_data_cantilever(W, x, t, Fmax, n_samples):
    """"""
    U_samples = np.zeros((2, n_samples))
    E_samples = generator_E_cantilever(n_samples)
    I_samples = generator_I_cantilever(n_samples)
    U_samples[0, :] = E_samples
    U_samples[1, :] = I_samples

    n_y = x.size
    data = np.zeros((n_y + 1, t.size, n_samples))
    for i in range(n_samples):
        U = U_samples[:, i]
        data[:n_y, :, i] = model_cantilever_beam(W, U, x, t, Fmax)
        data[-1, :, i] = W

    return data


if __name__ == '__main__':
    # Fixing the seed for the random number generators
    np.random.seed(seed=42)

    # Starting timer
    t0 = time.time()

    ####
    # Generate a dataset, plot trajectories, perform PCA on model outputs, then recover model outputs
    # and plot recovered trajectories
    n_Y = 10
    n_samples_U = 10
    x = np.linspace(0.1, 1., n_Y)
    t = np.linspace(0.1, 1., 10)
    n_W = 10
    n_samples_tot = n_samples_U * n_W

    W = np.zeros((1, n_W))
    W[0, :] = np.linspace(0.1, 1.0, n_W)

    Fmax = 1.5e9

    data = np.zeros((n_Y + 1, t.size, n_samples_tot))
    for i in range(n_W):
        data_i = generate_data_cantilever(W[:, i], x, t, Fmax, n_samples_U)
        data[:, :, (i * n_samples_U):((i + 1) * n_samples_U)] = data_i
    dataset = Dataset(data, n_Y)

    n_q = 10
    dataset.pca_on_Y(n_q)
    dataset.full_pca_on_X()
    recovered_X = dataset.recover_X(dataset.H_data)
    recovered_data = dataset.recover_data(recovered_X)

    _, ax = plt.subplots()
    for i in range(n_samples_tot):
        ax.plot(x, data[:n_Y, -1, i], '-b')
    ax.set_title('Realizations of random variable Y at last time-step')
    ax.set_xlabel('x')
    ax.set_ylabel('deflection')
    plt.grid()
    plt.savefig('./cantilever_original_data.png')

    _, ax = plt.subplots()
    for i in range(n_samples_tot):
        ax.plot(x, recovered_data[:n_Y, -1, i], '-b')
    ax.set_title('Recovered realizations of random variable Y at last time-step')
    ax.set_xlabel('x')
    ax.set_ylabel('deflection')
    plt.grid()
    plt.savefig('./cantilever_recovered_data.png')

    ####
    # Generate a large number of additional realizations from an original dataset
    # using diffusion maps basis and the ISDE generator

    # Diffusion Maps basis parameters
    eps = 3.
    m = 30
    kappa = 1
    # ISDE generator parameters
    s_nu = np.power(4 / (n_samples_tot * (2 + dataset.H_data.shape[0])), 1 / (dataset.H_data.shape[0] + 4))
    s_hat_nu = s_nu / (np.sqrt(s_nu ** 2 + ((n_samples_tot - 1) / n_samples_tot)))
    Fac = 20
    delta_r = 2 * np.pi * s_hat_nu / Fac
    f_0 = 1.5
    M_0 = 300
    n_MC = 20
    # Parallel processing parameters
    n_cpu = 4

    print('Generating additional realizations...')
    generator = Generator(dataset, n_cpu, Fac, delta_r, f_0, M_0, eps, kappa, m)
    generator.construct_dmaps_basis(plot_eigvals_name='cantilever')
    total_data_MCMC = generator.generate_realizations(n_MC)
    print(f'Number of additional realizations: {total_data_MCMC.shape[2]}')

    # Plot additional realizations
    _, ax = plt.subplots()
    for i in range(total_data_MCMC.shape[-1]):
        ax.plot(x, total_data_MCMC[:n_Y, -1, i], '-r')
    ax.set_title('Additional realizations of Y at last time-step')
    ax.set_xlabel('x')
    ax.set_ylabel('deflection')
    plt.grid()
    plt.savefig('./cantilever_mcmc_data.png')

    ####
    # Create a surrogate model for the last time-step, compute a conditional mean and confidence interval, plot results
    print('Computing surrogate model...')

    W_conditional = np.array([1.])
    surrogate_model = Surrogate(total_data_MCMC, n_Y, t)

    surrogate_n_samples = 10000
    confidence_level = 0.95

    surrogate_model.compute_surrogate_gkde(t.size - 1)
    surrogate_mean = surrogate_model.compute_conditional_mean(W_conditional, surrogate_n_samples)
    surrogate_lower_bound, surrogate_upper_bound = surrogate_model.compute_conditional_confidence_interval(
        W_conditional,
        surrogate_n_samples,
        confidence_level)

    # Plot
    _, ax = plt.subplots()
    ax.plot(x, surrogate_mean, '-k', label='mean')
    ax.plot(x, surrogate_lower_bound, '--g', label='lower confidence bound')
    ax.plot(x, surrogate_upper_bound, '--r', label='upper confidence bound')
    ax.fill_between(x, surrogate_lower_bound, surrogate_upper_bound, color='cyan')
    ax.set_title(f'Surrogate model conditional prediction of deflection\nat last time-step: '
                 f'mean and 95% confidence interval\nW={W_conditional[0]}')
    ax.set_xlabel('x')
    ax.set_ylabel('deflection')
    ax.legend()
    plt.grid()
    plt.savefig('./cantilever_surrogate_deflection.png')

    # Save surrogate model
    surrogate_model.save_surrogate('./cantilever_surrogate.dill')

    ####
    # Ending timer
    t1 = time.time()
    # Showing elapsed time
    total = t1 - t0
    print(f'Execution took {total:.2f} seconds.')