import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal
from scipy.linalg import solve
import dill


class GaussianKde(gaussian_kde):
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

    def conditional_resample(
            self,
            n_samples,
            cond_idx: list[int],
            cond_values: np.ndarray,
    ) -> np.ndarray:
        """
        Conditional sampling for gaussian kernel density estimation.

        Parameters
        ----------
        cond_idx : list[int]
            Indices of conditioning variables (ex: [2, 3] for X3 and X4).
        cond_values : np.ndarray
            Observed values for conditioning variables, of shape (len(cond_idx),).
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        np.ndarray
            Samples of the free variables, of shape (len(free_idx), n_samples).
        """
        d, n = self.dataset.shape
        cond_idx = list(cond_idx)
        free_idx = [i for i in range(d) if i not in cond_idx]

        H = self.covariance  # (d, d)

        # Covariance matrix blocks
        H_ff = H[np.ix_(free_idx, free_idx)]
        H_cc = H[np.ix_(cond_idx, cond_idx)]
        H_fc = H[np.ix_(free_idx, cond_idx)]

        # Conditional covariance
        A = solve(H_cc, H_fc.T, assume_a="pos")  # (len_cond, len_free)
        cov_cond = H_ff - H_fc @ A  # (len_free, len_free)

        B = A.T

        dataset_cond = self.dataset[cond_idx, :]  # (len_cond, n)
        residuals = cond_values[:, None] - dataset_cond  # (len_cond, n)

        # Log-density of each marginal component on the conditioning variables
        log_weights = multivariate_normal.logpdf(
            residuals.T,  # (n, len_cond)
            mean=np.zeros(len(cond_idx)),
            cov=H_cc,
        )
        # Numerical stabilisation and normalization
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()

        # --- Sample generation ---
        samples = np.empty((n_samples, len(free_idx)))
        dataset_free = self.dataset[free_idx, :]  # (len_free, n)

        chosen = np.random.choice(n, size=n_samples, p=weights)

        for k, i in enumerate(chosen):
            mu_free_i = dataset_free[:, i]
            delta = cond_values - dataset_cond[:, i]  # (len_cond,)
            mean_cond = mu_free_i + B @ delta

            samples[k] = np.random.multivariate_normal(mean_cond, cov_cond)

        return samples.T


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
                                                           cond_idx=range(self.n_Y, self.data.shape[0]),
                                                           cond_values=W)
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