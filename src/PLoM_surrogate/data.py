import numpy as np

from PLoM_surrogate.generators import generator_U
from PLoM_surrogate.models import model_sinc


def generate_data_sinc(W, t, n_samples):
    """

    Parameters
    ----------
    W: 2-dimensional vector of control parameters
    t: List or numpy vector of Nt time values for which the output is calculated
    n_samples: Number of desired sample

    Returns
    -------
    dataset: 3xNtxn_samples numpy array of realisations of the couple (Y, W). That is, along the first axis, the first
    component is a realization of Y (output of the model) at a given timestep, and the next two components are
    the values of the control parameters used to generate the Y time series data.

    """
    U_samples = generator_U(n_samples)
    data = np.zeros((3, t.size, n_samples))
    for i in range(n_samples):
        U = U_samples[:, i]
        data[0, :, i] = model_sinc(W, U, t)
    data[1:, :, :] = np.tile(W, (1, t.size, n_samples))

    return data


class Dataset:
    """
    Class to store and reduce data at different steps of the methodology.
    """
    def __init__(self, data):
        # first dimension is the components (outputs of the model and control parameters),
        # second dimension is the timesteps, third dimension is the realizations
        self.data = data

        self.vecs_mean_data = []
        self.mats_V_data = []
        self.X_data = None

        self.vec_mean_X = None
        self.mat_V_X = None
        self.H_data = None

    def full_pca_on_Y(self):
        """"""

    def recover_data(self, X):
        data = np.zeros((self.vecs_mean_data[0].size, self.data.shape[1], X.shape[1]))
        for i in range(self.data.shape[1]):
            vec_mean_data = self.vecs_mean_data[i]
            mat_V_data = self.mats_V_data[i]
            data[:, i, :] = np.dot(mat_V_data, X) + np.tile(vec_mean_data, (1, X.shape[1]))

    def full_pca_on_X(self):
        """"""

    def recover_X(self, H):
        """"""
        X = np.dot(self.mat_V_X, H) + np.tile(self.vec_mean_X, (1, H.shape[1]))
        return X
