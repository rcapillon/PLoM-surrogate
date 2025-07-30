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
    def __init__(self, data, n_Y):
        # first dimension is the components (n_Y outputs of the model and a number of control parameters),
        # second dimension is the timesteps, third dimension is the realizations
        self.data = data
        self.n_Y = n_Y
        self.dim = data.shape[0]
        self.n_t = data.shape[1]
        self.n_r = data.shape[2]

        self.vecs_mean_Y = []
        self.mats_V_Y = []
        self.X_data = None

        self.vec_mean_X = None
        self.mat_V_X = None
        self.H_data = None

    def full_pca_on_Y(self):
        """
        Performs a full-order PCA on the model outputs' realizations of time-series
        """
        Y_data = self.data[:self.n_Y, :, :]
        self.X_data = np.zeros((self.dim, self.n_r))
        self.X_data[self.n_Y:, :] = self.data[self.n_Y:, 0, :]
        Q_data = np.zeros((self.n_Y, self.n_r))

        for i in range(self.n_t):
            mat_mean_covar = np.zeros((self.n_Y, self.n_Y))

            vec_mean_Y_i = np.mean(Y_data[:, i, :], axis=-1)
            self.vecs_mean_Y.append(vec_mean_Y_i)
            centered_Y_t_data_i = Y_data[:, i, :] - np.tile(vec_mean_Y_i[:, np.newaxis], (1, self.n_r))
            for j in range(self.n_t):
                centered_Y_t_data_j = Y_data[:, j, :] - np.tile(vec_mean_Y_i[:, np.newaxis], (1, self.n_r))
                mat_mean_covar += np.dot(centered_Y_t_data_i, centered_Y_t_data_j.T) / (self.n_r - 1)
            mat_mean_covar /= self.n_t

            eigvals, eigvects = np.linalg.eig(mat_mean_covar)
            inds_sort_eig = np.flip(np.argsort(eigvals))
            sorted_eigvals = eigvals[inds_sort_eig]
            sorted_eigvects = eigvects[:, inds_sort_eig]

            mat_V_Y_i = np.zeros((self.n_Y, self.n_Y))
            for j in range(self.n_Y):
                mat_V_Y_i[:, j] = np.sqrt(sorted_eigvals[j]) * sorted_eigvects[:, j]
            self.mats_V_Y.append(mat_V_Y_i)

            Q_data += np.dot(np.diag(1 / sorted_eigvals), np.dot(mat_V_Y_i.T, centered_Y_t_data_i))
        Q_data /= self.n_t

        self.X_data[:self.n_Y, :] = Q_data

    def recover_data(self, X):
        """
        Recovers model outputs' realization of time series from its full-order PCA transformation
        """
        data = np.zeros((self.dim, self.n_t, X.shape[1]))
        data[self.n_Y:, :, :] = X[self.n_Y:, :, :]
        for i in range(self.n_t):
            vec_mean_Y = self.vecs_mean_Y[i]
            mat_V_Y = self.mats_V_Y[i]
            data[:self.n_Y, i, :] = np.dot(mat_V_Y, X) + np.tile(vec_mean_Y[:, np.newaxis], (1, X.shape[1]))

        return data

    def full_pca_on_X(self):
        """
        Performs a full-order PCA on the 'X' dataset, where full-order PCA has already been applied to the model outputs
        """
        self.H_data = np.zeros((self.dim, self.n_r))

        self.vec_mean_X = np.mean(self.X_data, axis=-1)
        centered_X_data = self.X_data - np.tile(self.vec_mean_X[:, np.newaxis], (1, self.n_r))
        mat_covar = np.dot(centered_X_data, centered_X_data.T) / (self.n_r - 1)

        eigvals, eigvects = np.linalg.eig(mat_covar)
        inds_sort_eig = np.flip(np.argsort(eigvals))
        sorted_eigvals = eigvals[inds_sort_eig]
        sorted_eigvects = eigvects[:, inds_sort_eig]

        self.mat_V_X = np.dot(sorted_eigvects, np.diag(np.sqrt(sorted_eigvals)))

        self.H_data = np.dot(np.diag(1 / np.sqrt(sorted_eigvals)), np.dot(sorted_eigvects.T, centered_X_data))

    def recover_X(self, H):
        """
        Recovers X data (where model outputs are still transformed by their own PCA)
        from its full-order PCA transformation
        """
        X = np.dot(self.mat_V_X, H) + np.tile(self.vec_mean_X[:, np.newaxis], (1, H.shape[1]))

        return X
