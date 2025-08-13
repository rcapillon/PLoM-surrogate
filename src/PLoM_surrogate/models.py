from typing import Union
import numpy as np
from scipy.stats import gaussian_kde
from kdetools import gaussian_kde as gkde
import dill


class GaussianKde(gkde):
    # source: https://stackoverflow.com/questions/63812970/scipy-gaussian-kde-matrix-is-not-positive-definite
    """
    Drop-in replacement for gaussian_kde that adds the class attribute EPSILON
    to the covmat eigenvalues, to prevent exceptions due to numerical error.
    """

    EPSILON = 1e-10  # adjust this at will

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                                         bias=False,
                                                         aweights=self.weights))
            # we're going the easy way here
            self._data_covariance += self.EPSILON * np.eye(
                len(self._data_covariance))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor ** 2
        # self.inv_cov = self._data_inv_cov / self.factor ** 2
        L = np.linalg.cholesky(self.covariance * 2 * np.pi)
        self._norm_factor = 2 * np.log(np.diag(L)).sum()  # needed for scipy 1.5.2
        self.log_det = 2 * np.log(np.diag(L)).sum()  # changed var name on 1.6.2


class Surrogate:
    """
    Surrogate model for a probabilistic model parametrized by control parameters. Can be used to generate samples,
    or estimate lower and upper confidence bounds, for a given value of time (or pseudo-time e.g. frequency)
    and for given values of the control parameters.
    """
    def __init__(self, data, n_Y):
        self.data = data
        self.n_Y = n_Y

        self.idx_t = None
        self.surrogate_gkde = None
        self.conditional_marginal_pdf_gkde = None

    def compute_surrogate_gkde(self, idx_t):
        """"""
        self.idx_t = idx_t
        self.surrogate_gkde = GaussianKde(self.data[:, idx_t, :], bw_method='silverman')

    def sample(self, n_samples):
        """"""
        samples = self.surrogate_gkde.resample(n_samples)

        return samples

    def conditional_sample(self, W, n_samples):
        """"""
        samples = self.surrogate_gkde.conditional_resample(n_samples,
                                                           x_cond=W[:, np.newaxis].transpose(),
                                                           dims_cond=list(range(self.n_Y, self.data.shape[0])))
        samples = np.squeeze(samples, axis=0).transpose()
        return samples

    def compute_conditional_mean(self, W, n_samples):
        """"""
        samples = self.conditional_sample(W, n_samples)
        mean = np.mean(samples, axis=1)

        return mean

    def compute_conditional_covar(self, W, n_samples):
        """"""
        samples = self.conditional_sample(W, n_samples)
        mean = np.mean(samples, axis=1)
        centered_samples = samples - np.tile(mean[:, np.newaxis], (1, n_samples))
        covar = np.dot(centered_samples, centered_samples.T) / (n_samples - 1)

        return covar

    def compute_conditional_confidence_interval(self, W, n_samples, p_confidence=0.95):
        """"""
        n_rejected_samples = int(np.floor((1 - p_confidence) * n_samples / 2))
        Y_lower_confidence_bound = np.zeros((self.n_Y,))
        Y_upper_confidence_bound = np.zeros((self.n_Y,))

        samples = self.conditional_sample(W, n_samples)
        for i in range(self.n_Y):
            ordered_samples_i = np.sort(samples[i, :])
            Y_lower_confidence_bound[i] = ordered_samples_i[n_rejected_samples]
            Y_upper_confidence_bound[i] = ordered_samples_i[-(n_rejected_samples + 1)]

        return Y_lower_confidence_bound, Y_upper_confidence_bound

    def evaluate_conditional_marginal_pdf(self, idx_y, W, n_samples, ymin, ymax, n_points):
        samples = self.conditional_sample(W, n_samples)
        self.conditional_marginal_pdf_gkde = gaussian_kde(samples[idx_y, :])
        points = np.linspace(ymin, ymax, n_points)
        pdf_values = self.conditional_marginal_pdf_gkde.pdf(points)

        return pdf_values

    def save_surrogate(self, save_path):
        """
        Saves the Surrogate object to a dill file.

        Parameters
        ----------
        save_path: path to the file where the surrogate will be saved

        """
        with open(save_path, 'wb') as file:
            dill.dump(self, file)


def load_surrogate(load_path):
    """
    Loads a Surrogate object saved in a dill file.

    Parameters
    ----------
    load_path: path to the file where the surrogate is saved

    Returns
    -------
    surrogate: Surrogate object

    """
    with open(load_path, 'rb') as file:
        surrogate = dill.load(file)

    return surrogate