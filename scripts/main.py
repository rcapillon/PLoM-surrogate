import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import gaussian_kde
from PLoM_surrogate.models import model_sinc
from PLoM_surrogate.generators import generator_U, generator_mat_N, generator_delta_Wiener
from PLoM_surrogate.data import generate_data_sinc, Dataset


if __name__ == '__main__':
    # Fixing the seed for the random number generators
    np.random.seed(seed=42)

    # # Estimation of the covariance matrix of U
    #
    # n_U_samples = 10000
    # U_samples = generator_U(n_U_samples)
    #
    # mean_U = np.mean(U_samples, axis=-1)
    # centered_U_samples = U_samples - np.tile(mean_U[:, np.newaxis], (1, n_U_samples))
    # cov_U = np.dot(centered_U_samples, centered_U_samples.T) / (n_U_samples - 1)
    # print(cov_U)

    # Generate a dataset, plot trajectories, perform PCA on model outputs, then recover model outputs
    # and plot recovered trajectories
    n_Y = 1
    n_samples_U = 20
    t = np.linspace(0., 10 * np.pi, 100)
    n_W = 5
    n_W_tot = n_W ** 2
    n_samples_tot = n_samples_U * n_W_tot

    W0 = np.linspace(1., 3., n_W)
    W1 = np.linspace(0., 2., n_W)
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

    n_q = 20
    dataset.pca_on_Y(n_q)
    # recovered_data = dataset.recover_data(dataset.X_data)
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
