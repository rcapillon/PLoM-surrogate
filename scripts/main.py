import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import gaussian_kde
from PLoM_surrogate.models import model_sinc
from PLoM_surrogate.generators import generator_U

if __name__ == '__main__':
    n_U_samples = 100000
    U_samples = generator_U(n_U_samples)
    n_Y_samples = 100
    W = np.array([2., 1.])
    mat_U = U_samples[:, :n_Y_samples]
    t = np.linspace(0., 10 * np.pi, 100)
    rand_Y = np.zeros((n_Y_samples, len(t)))
    for i in range(n_Y_samples):
        U = mat_U[:, i]
        rand_Y[i, :] = model_sinc(W, U, t)

    # U0_gkde = gaussian_kde(U_samples[0, :])
    # U1_gkde = gaussian_kde(U_samples[1, :])
    # x_U0 = np.linspace(-2., 5., 1000)
    # x_U1 = np.linspace(5., 8., 1000)
    # pdf_U0 = U0_gkde.pdf(x_U0)
    # pdf_U1 = U1_gkde.pdf(x_U1)
    #
    # _, ax = plt.subplots()
    # ax.plot(x_U0, pdf_U0, '-b', label='U_0 pdf')
    # ax.set_title('Marginal PDF of U_0')
    # ax.set_xlabel('x')
    # ax.set_ylabel('p(x)')
    # plt.grid()
    # plt.show()
    #
    # _, ax = plt.subplots()
    # ax.plot(x_U1, pdf_U1, '-r', label='U_1 pdf')
    # ax.set_title('Marginal PDF of U_1')
    # ax.set_xlabel('x')
    # ax.set_ylabel('p(x)')
    # plt.grid()
    # plt.show()

    _, ax = plt.subplots()
    for i in range(n_Y_samples):
        ax.plot(t, rand_Y[i, :], '-b')
    ax.set_title('Trajectories of random variable Y')
    ax.set_xlabel('t')
    ax.set_ylabel('Y')
    plt.grid()
    plt.show()