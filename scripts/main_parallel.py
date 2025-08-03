import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

from PLoM_surrogate.models import model_sinc
from PLoM_surrogate.generators import generator_U, generator_ISDE
from PLoM_surrogate.data import generate_data_sinc, Dataset
from PLoM_surrogate.dmaps import construct_dmaps_basis, build_mat_a


if __name__ == '__main__':
    # Fixing the seed for the random number generators
    np.random.seed(seed=42)

    # Starting timer
    t0 = time.time()

    # Generate a dataset, plot trajectories, perform PCA on model outputs, then recover model outputs
    # and plot recovered trajectories
    n_Y = 1
    n_samples_U = 10
    t = np.linspace(0., 10 * np.pi, 40)
    n_W = 5
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
    plt.show()

    _, ax = plt.subplots()
    for i in range(n_samples_tot):
        ax.plot(t, recovered_data[0, :, i], '-b')
    ax.set_title('Trajectories of recovered random variable Y')
    ax.set_xlabel('t')
    ax.set_ylabel('Y')
    plt.grid()
    plt.show()

    # Generate a large number of additional realizations from an original dataset
    # using diffusion maps basis and the ISDE generator

    s_nu = np.power(4 / (n_samples_tot * (2 + dataset.H_data.shape[0])), 1 / (dataset.H_data.shape[0] + 4))
    s_hat_nu = s_nu / (np.sqrt(s_nu ** 2 + ((n_samples_tot - 1) / n_samples_tot)))
    Fac = 20
    delta_r = 2 * np.pi * s_hat_nu / Fac
    f_0 = 1.5
    M_0 = 200
    n_MC = 100

    eps = 3.
    m = 10
    kappa = 1
    mat_g = construct_dmaps_basis(dataset.H_data, eps, m, kappa)
    mat_a = build_mat_a(mat_g)

    # parallel processing
    n_cpu = 6
    n_jobs_per_cpu = 1
    pool = Pool(processes=n_cpu)

    total_data_MCMC = np.empty((n_Y + W.shape[0], t.size, 0))
    progress_bar = True
    inputs = [(dataset, mat_a, mat_g, delta_r, f_0, M_0, n_MC, progress_bar)] * n_cpu * n_jobs_per_cpu

    for data_MCMC in pool.starmap(generator_ISDE, inputs):
        total_data_MCMC = np.concatenate((total_data_MCMC, data_MCMC), axis=-1)

    _, ax = plt.subplots()
    for i in range(n_samples_tot):
        ax.plot(t, total_data_MCMC[0, :, i], '-r')
    ax.set_title('Trajectories of additional realizations of Y')
    ax.set_xlabel('t')
    ax.set_ylabel('Y')
    plt.grid()
    plt.show()

    # Ending timer
    t1 = time.time()
    # Showing elapsed time
    total = t1 - t0
    print(f'Execution took {total:.2f} seconds.')
