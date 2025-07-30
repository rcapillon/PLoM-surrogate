import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import gaussian_kde
from PLoM_surrogate.models import model_sinc
from PLoM_surrogate.generators import generator_U

if __name__ == '__main__':
    # W = np.array([1., 0.])
    # U = np.array([0., 2 * np.pi])
    # t = np.linspace(0., 10 * np.pi, 100)
    # W = np.array([1., 0.5])
    # U = np.array([0.5, 2.5 * np.pi])
    # t = np.linspace(0., 10 * np.pi, 100)
    # Y = model_sinc(W, U, t)
    #
    # _, ax = plt.subplots()
    # ax.plot(t, Y, '-b', label='Y(t;w,u)')
    # ax.set_title('Graph of model output')
    # ax.set_xlabel('t')
    # ax.set_ylabel('Y')
    # plt.grid()
    # plt.legend()
    # plt.show()
    # plt.savefig('./test.png')

    n_samples = 1e4
    U = generator_U(n_samples)
    print(U.shape)

    # U0_gkde = gaussian_kde(U[0, :])
    # U1_gkde = gaussian_kde(U[1, :])
    #
    # x_U0 = np.linspace(4., 8., 100)
    # x_U1 = np.linspace(4., 7., 100)
    #
    # pdf_U0 = U0_gkde.pdf(x_U0)
    # pdf_U1 = U1_gkde.pdf(x_U1)
    #
    # _, ax = plt.subplots()
    # ax.plot(x_U0, pdf_U0, '-b', label='U0 pdf')
    # ax.plot(x_U1, pdf_U1, '-r', label='U0 pdf')
    # ax.set_title('PDFs')
    # ax.set_xlabel('x')
    # ax.set_ylabel('p(x)')
    # plt.grid()
    # plt.legend()
    # plt.show()
