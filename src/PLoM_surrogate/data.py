import numpy as np

from PLoM_surrogate.generators import generator_U_sinc
from PLoM_surrogate.models import model_sinc
from numpy.ma.core import zeros_like


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

        self.min_data = np.zeros((self.dim, self.n_t))
        self.max_data = np.zeros((self.dim, self.n_t))

        self.n_q = None

        self.vec_mean_Y = None
        self.vec_eig_Y = None
        self.mat_phi_Y = None
        self.X_data = None

        self.vec_mean_X = None
        self.vec_eig_X = None
        self.mat_phi_X = None
        self.H_data = None

    def scale_data_wrt_realizations(self):
        scaled_data = np.zeros((self.dim, self.n_t, self.n_r))
        for i in range(self.dim):
            for j in range(self.n_t):
                self.min_data[i, j] = np.amin(self.data[i, j, :])
                self.max_data[i, j] = np.amax(self.data[i, j, :])
                scaled_data[i, j, :] = (1e-10 + (self.data[i, j, :] - self.min_data[i, j])
                                        / (self.max_data[i,j] - self.min_data[i, j]))

        return scaled_data

    def unscale_data_wrt_realizations(self, scaled_data):
        unscaled_data = np.zeros((self.dim, self.n_t, scaled_data.shape[2]))
        for i in range(self.dim):
            for j in range(self.n_t):
                unscaled_data[i, j, :] = ((scaled_data[i, j, :] - 1e-10) * (self.max_data[i, j] - self.min_data[i, j])
                                          + self.min_data[i, j])

        return unscaled_data

    def pca_on_Y(self, n_q):
        """
        Performs a PCA on the model outputs' realizations of time-series
        """
        self.n_q = n_q

        scaled_data = self.scale_data_wrt_realizations()

        reshaped_Y_data = np.zeros((self.n_Y * self.n_t, self.n_r))
        for i in range(self.n_Y):
            reshaped_Y_data[(i * self.n_t):(i * self.n_t + self.n_t), :] = scaled_data[i, :, :]

        self.vec_mean_Y = np.mean(reshaped_Y_data, axis=-1)
        centered_Y_data = reshaped_Y_data - np.tile(self.vec_mean_Y[:, np.newaxis], (1, self.n_r))

        self.X_data = np.zeros((n_q + self.dim - self.n_Y, self.n_r))
        self.X_data[n_q:, :] = scaled_data[self.n_Y:, 0, :]

        b, S, _ = np.linalg.svd(centered_Y_data, full_matrices=False)
        inds_sort_sv = np.flip(np.argsort(S))
        sorted_sv = S[inds_sort_sv]
        sorted_b = b[:, inds_sort_sv]

        self.vec_eig_Y = sorted_sv[:self.n_q] ** 2 / (self.n_r - 1)
        self.mat_phi_Y = sorted_b[:, :self.n_q]

        self.X_data[:self.n_q, :] = np.linalg.solve(np.diag(np.sqrt(self.vec_eig_Y)),
                                                    np.dot(self.mat_phi_Y.T, centered_Y_data))

    def recover_data(self, X):
        """
        Recovers model outputs' realization of time series from its PCA transformation
        """
        recovered_reshaped_data = (np.dot(np.dot(self.mat_phi_Y, np.diag(np.sqrt(self.vec_eig_Y))), X[:self.n_q, :])
                                   + np.tile(self.vec_mean_Y[:, np.newaxis], (1, X.shape[1])))

        recovered_data = np.zeros((self.dim, self.n_t, X.shape[1]))
        recovered_data[self.n_Y:, :, :] = np.tile(X[self.n_q:, np.newaxis, :], (1, self.n_t, 1))

        for i in range(self.n_Y):
            recovered_data[i, :, :] = recovered_reshaped_data[(i + i * self.n_t):(i + (i + 1) * self.n_t), :]

        recovered_unscaled_data = self.unscale_data_wrt_realizations(recovered_data)

        return recovered_unscaled_data

    def full_pca_on_X(self):
        """
        Performs a full-order PCA on the 'X' dataset, where PCA has already been applied to the model outputs
        """
        self.H_data = np.zeros((self.dim, self.n_r))

        self.vec_mean_X = np.mean(self.X_data, axis=-1)
        centered_X_data = self.X_data - np.tile(self.vec_mean_X[:, np.newaxis], (1, self.n_r))

        mat_covar = np.dot(centered_X_data, centered_X_data.T) / (self.n_r - 1)

        eigvals, eigvects = np.linalg.eig(mat_covar)

        inds_keep = eigvals > 1e-9
        eigvals = eigvals[inds_keep]
        eigvects = eigvects[:, inds_keep]

        inds_sort_eig = np.flip(np.argsort(eigvals))
        self.vec_eig_X = eigvals[inds_sort_eig]
        self.mat_phi_X = eigvects[:, inds_sort_eig]

        self.H_data = np.dot(np.diag(1 / np.sqrt(self.vec_eig_X)), np.dot(self.mat_phi_X.T, centered_X_data))

    def recover_X(self, H):
        """
        Recovers X data (where model outputs are still transformed by their own PCA)
        from its full-order PCA transformation
        """
        recovered_X = (np.dot(np.dot(self.mat_phi_X, np.diag(np.sqrt(self.vec_eig_X))), H)
                       + np.tile(self.vec_mean_X[:, np.newaxis], (1, H.shape[1])))

        return recovered_X
