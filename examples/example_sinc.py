from typing import Union
import numpy as np
from scipy.stats import beta, truncnorm
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from src.PLoM_surrogate.data import Dataset
from src.PLoM_surrogate.generators import Generator
from src.PLoM_surrogate.models import Surrogate


def model_sinc(W: np.ndarray, U: np.ndarray, t: Union[list, np.ndarray]):
    """
    Simple model generating a scalar time series from 2 control and 2 uncertain parameters.

    Parameters
    ----------
    W: 2-dimensional vector of control parameters
    U: 2-dimensional vector of uncertain parameters
    t: List or numpy vector of Nt time values for which the output is calculated

    Returns
    -------
    Y: Nt-dimensional vector of outputs (time series)

    """
    if W.ndim != 1 or W.size != 2:
        raise ValueError('W must be a numpy vector with 2 components.')
    if U.ndim != 1 or U.size != 2:
        raise ValueError('U must be a numpy vector with 2 components.')
    if isinstance(t, list):
        arr_t = np.array(t)
    elif isinstance(t, np.ndarray) and t.ndim == 1:
        arr_t = t
    else:
        raise TypeError('Argument t must be a list or a numpy vector.')

    Y = W[0] * np.sinc((2 * arr_t + U[0]) / U[1]) + W[1]

    return Y


def generator_U_sinc(n_samples):
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


def generate_data_sinc(W, t, n_samples):
    """

    Parameters
    ----------
    W: 2-dimensional numpy vector of control parameter values
    t: List or numpy vector of Nt time values for which the output is calculated
    n_samples: Number of desired sample

    Returns
    -------
    dataset: 3xNtxn_samples numpy array of realisations of the couple (Y, W). That is, along the first axis, the first
    component is a realization of Y (output of the model) at a given timestep, and the next two components are
    the values of the control parameters used to generate the Y time series data.

    """
    U_samples = generator_U_sinc(n_samples)
    data = np.zeros((3, t.size, n_samples))
    for i in range(n_samples):
        U = U_samples[:, i]
        data[0, :, i] = model_sinc(W, U, t)
        data[1:, :, i] = np.tile(W[:, np.newaxis], (1, t.size))

    return data


if __name__ == '__main__':
    # Fixing the seed for the random number generators
    np.random.seed(seed=42)

    # Starting timer
    t0 = time.time()

    ####
    # Generate a dataset, plot trajectories, perform PCA on model outputs, then recover model outputs
    # and plot recovered trajectories
    n_Y = 1
    n_samples_U = 10
    t = np.linspace(0., 10 * np.pi, 100)
    n_W = 3
    n_W_tot = n_W ** 2
    n_samples_tot = n_samples_U * n_W_tot

    W0 = np.linspace(1.5, 2.5, n_W)
    W1 = np.linspace(0.5, 1.5, n_W)
    W = np.zeros((2, n_W_tot))
    counter = 0
    for i in range(n_W):
        w0 = W0[i]
        for j in range(n_W):
            w1 = W1[j]
            W[0, counter] = w0
            W[1, counter] = w1
            counter += 1

    data = np.zeros((3, t.size, n_samples_tot))
    for i in range(n_W_tot):
        data_i = generate_data_sinc(W[:, i], t, n_samples_U)
        data[:, :, (i * n_samples_U):((i + 1) * n_samples_U)] = data_i
    dataset = Dataset(data, n_Y)

    n_q = 5
    dataset.pca_on_Y(n_q)
    dataset.full_pca_on_X()
    recovered_X = dataset.recover_X(dataset.H_data)
    recovered_data = dataset.recover_data(recovered_X)

    _, ax = plt.subplots()
    for i in range(n_samples_tot):
        ax.plot(t, data[0, :, i], '-b')
    ax.set_title('Trajectories of random variable Y')
    ax.set_xlabel('t')
    ax.set_ylabel('Y')
    plt.grid()
    plt.savefig('./sinc_original_data.png')

    _, ax = plt.subplots()
    for i in range(n_samples_tot):
        ax.plot(t, recovered_data[0, :, i], '-b')
    ax.set_title('Trajectories of recovered random variable Y')
    ax.set_xlabel('t')
    ax.set_ylabel('Y')
    plt.grid()
    plt.savefig('./sinc_recovered_data.png')

    ####
    # Generate a large number of additional realizations from an original dataset
    # using diffusion maps basis and the ISDE generator

    # Diffusion Maps basis parameters
    eps = 3.
    m = 70
    kappa = 1
    # ISDE generator parameters
    s_nu = np.power(4 / (n_samples_tot * (2 + dataset.H_data.shape[0])), 1 / (dataset.H_data.shape[0] + 4))
    s_hat_nu = s_nu / (np.sqrt(s_nu ** 2 + ((n_samples_tot - 1) / n_samples_tot)))
    Fac = 20
    delta_r = 2 * np.pi * s_hat_nu / Fac
    f_0 = 1.5
    M_0 = 100
    n_MC = 60
    # Parallel processing parameters
    n_cpu = 4

    print('Generating additional realizations...')
    generator = Generator(dataset, n_cpu, Fac, delta_r, f_0, M_0, eps, kappa, m)
    generator.construct_dmaps_basis(plot_eigvals_name='cantilever')
    total_data_MCMC = generator.generate_realizations(n_MC)
    print(f'Number of additional realizations: {total_data_MCMC.shape[2]}')

    _, ax = plt.subplots()
    for i in range(n_samples_tot):
        ax.plot(t, total_data_MCMC[0, :, i], '-r')
    ax.set_title('Trajectories of additional realizations of Y')
    ax.set_xlabel('t')
    ax.set_ylabel('Y')
    plt.grid()
    plt.savefig('./sinc_mcmc_data.png')

    ####
    # Create a surrogate model for every time-step, compute a conditional mean and confidence interval, plot results
    print('Computing surrogate model...')

    W_conditional = np.array([2.25, 0.75])
    surrogate_model = Surrogate(total_data_MCMC, n_Y)

    surrogate_n_samples = 10000
    confidence_level = 0.99
    ls_surrogate_mean = []
    ls_surrogate_lower_bound = np.zeros((t.size, ))
    ls_surrogate_upper_bound = np.zeros((t.size, ))

    for i in tqdm(range(t.size)):
        surrogate_model.compute_surrogate_gkde(i)
        mean_i = surrogate_model.compute_conditional_mean(W_conditional, surrogate_n_samples)
        ls_surrogate_mean.append(mean_i)
        lower_bound, upper_bound = surrogate_model.compute_conditional_confidence_interval(W_conditional,
                                                                                           surrogate_n_samples,
                                                                                           confidence_level)
        ls_surrogate_lower_bound[i] = lower_bound[0]
        ls_surrogate_upper_bound[i] = upper_bound[0]

    # Plot conditional mean trajectory with confidence interval
    _, ax = plt.subplots()
    ax.plot(t, ls_surrogate_mean, '-k', label='mean')
    ax.plot(t, ls_surrogate_lower_bound, '--g', label='lower confidence bound')
    ax.plot(t, ls_surrogate_upper_bound, '--r', label='upper confidence bound')
    ax.fill_between(t, ls_surrogate_lower_bound, ls_surrogate_upper_bound, color='cyan')
    ax.set_title(f'Surrogate model conditional prediction: mean and 95% confidence interval\n'
                 f'W0={W_conditional[0]} , W1= {W_conditional[1]}')
    ax.set_xlabel('t')
    ax.set_ylabel('Y')
    ax.legend()
    plt.grid()
    plt.savefig('./sinc_surrogate_timeseries.png')

    # Plot conditional pdf at given time-step
    idx_y = 0
    idx_t = 20
    ymin = 0.5
    ymax = 1.5
    n_points = 1000
    points = np.linspace(ymin, ymax, n_points)
    surrogate_model.compute_surrogate_gkde(idx_t)
    pdf_values = surrogate_model.evaluate_conditional_marginal_pdf(idx_y, W_conditional,
                                                                   surrogate_n_samples, ymin, ymax, n_points)

    _, ax = plt.subplots()
    ax.plot(points, pdf_values, '-b')
    ax.set_title(f'Probability Density Function at {idx_t}-th time step')
    ax.set_xlabel('y')
    ax.set_ylabel('pdf')
    plt.grid()
    plt.savefig('./sinc_surrogate_pdf.png')

    # Saving surrogate model
    surrogate_model.save_surrogate('./sinc_surrogate.dill')

    # Ending timer
    t1 = time.time()
    # Showing elapsed time
    total = t1 - t0
    print(f'Execution took {total:.2f} seconds.')
