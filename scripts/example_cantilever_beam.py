import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from tqdm import tqdm

from PLoM_surrogate.models import model_sinc, Surrogate
from PLoM_surrogate.generators import generator_ISDE
from PLoM_surrogate.data import generate_data_cantilever, Dataset
from PLoM_surrogate.dmaps import construct_dmaps_basis, build_mat_a


if __name__ == '__main__':
    # Fixing the seed for the random number generators
    np.random.seed(seed=42)

    # Starting timer
    t0 = time.time()

    ####
    # Generate a dataset, plot trajectories, perform PCA on model outputs, then recover model outputs
    # and plot recovered trajectories
    n_Y = 10
    n_samples_U = 20
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

    s_nu = np.power(4 / (n_samples_tot * (2 + dataset.H_data.shape[0])), 1 / (dataset.H_data.shape[0] + 4))
    s_hat_nu = s_nu / (np.sqrt(s_nu ** 2 + ((n_samples_tot - 1) / n_samples_tot)))
    Fac = 20
    delta_r = 2 * np.pi * s_hat_nu / Fac
    f_0 = 1.5
    M_0 = 300
    n_MC = 30

    eps = 0.5
    # m = 125
    m = 30
    kappa = 1
    mat_g = construct_dmaps_basis(dataset.H_data, eps, m, kappa, plot_eigvals_name='cantilever')
    mat_a = build_mat_a(mat_g)

    # Parallel processing
    n_cpu = 4
    pool = Pool(processes=n_cpu)

    # MCMC
    total_data_MCMC = np.empty((n_Y + W.shape[0], t.size, 0))
    progress_bar = True
    inputs = [(dataset, mat_a, mat_g, delta_r, f_0, M_0, n_MC, progress_bar)] * n_cpu

    for data_MCMC in pool.starmap(generator_ISDE, inputs):
        # indices_delete = []
        # for i in range(data_MCMC.shape[-1]):
        #     if np.any(data_MCMC[:, :, i] >= 1e-3):
        #         indices_delete.append(i)
        # indices_keep = [i for i in range(data_MCMC.shape[-1]) if i not in indices_delete]
        # data_MCMC = data_MCMC[:, :, indices_keep]
        total_data_MCMC = np.concatenate((total_data_MCMC, data_MCMC), axis=-1)
    print(f'Number of additional realizations: {total_data_MCMC.shape[2]}')

    _, ax = plt.subplots()
    for i in range(total_data_MCMC.shape[-1]):
        ax.plot(x, total_data_MCMC[:n_Y, -1, i], '-r')
    ax.set_title('Additional realizations of Y at last time-step')
    ax.set_xlabel('x')
    ax.set_ylabel('deflection')
    plt.grid()
    plt.savefig('./cantilever_mcmc_data.png')

    ####
    # Create a surrogate model for every time-step, compute a conditional mean and confidence interval, plot results
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

    # Saving surrogate model
    surrogate_model.save_surrogate('./cantilever_surrogate.dill')

    # Ending timer
    t1 = time.time()
    # Showing elapsed time
    total = t1 - t0
    print(f'Execution took {total:.2f} seconds.')