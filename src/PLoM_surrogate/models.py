from typing import Union
import numpy as np
from scipy.special import erf
from kdetools import gaussian_kde as gkde
import dill


def model_sinc(W: np.ndarray, U: np.ndarray, t: Union[list, np.ndarray]):
    """
    Simple model generating a scalar time series from 2 control and 2 uncertain parameters.

    Parameters
    ----------
    W: 2-dimensional vector of control parameters
    U: 2-dimensional vector of uncertain parameters
    t: List or numpy vector of Nt time values for which the output is calculated

    Returns
    -------
    Y: Nt-dimensional vector of outputs (time series)

    """
    if W.ndim != 1 or W.size != 2:
        raise ValueError('W must be a numpy vector with 2 components.')
    if U.ndim != 1 or U.size != 2:
        raise ValueError('U must be a numpy vector with 2 components.')
    if isinstance(t, list):
        arr_t = np.array(t)
    elif isinstance(t, np.ndarray) and t.ndim == 1:
        arr_t = t
    else:
        raise TypeError('Argument t must be a list or a numpy vector.')

    Y = W[0] * np.sinc((2 * arr_t + U[0]) / U[1]) + W[1]

    return Y


def model_cantilever_beam(W: np.ndarray, U: np.ndarray,
                          x: Union[list, np.ndarray], t: Union[list, np.ndarray],
                          Fmax: float):
    """
    Quasi-static deflection of a cantilever beam of length 1m, subjected to a downwards concentrated load
    at a given abscissa.
    Source: https://home.engineering.iastate.edu/~shermanp/STAT447/STAT%20Articles/Beam_Deflection_Formulae.pdf

    Parameters
    ----------
    W: Single control parameter, corresponding to the abscissa where the load is applied
    U: 2-dimensional vector of uncertain parameters.
        U[0] is the Young's modulus, U[1] is the second moment of area (m^4)
    x: Abscissa along the beam
    t: List or numpy vector of Nt pseudo-time values between 0 and 1 for which the output is calculated
    Fmax: Constant parameter, corresponding to the maximal value for the load.
        The load evolves linearly with t, reaching Fmax at t=1

    Returns
    -------
    y: Nx x Nt array of deflections for the beam

    """
    if W.ndim != 1 or W.size != 1:
        raise ValueError('W must be a numpy vector with 1 components.')
    if U.ndim != 1 or U.size != 2:
        raise ValueError('U must be a numpy vector with 2 components.')
    if isinstance(x, list):
        arr_x = np.array(x)
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        arr_x = x
    else:
        raise TypeError('Argument x must be a list or a numpy vector.')
    if isinstance(t, list):
        arr_t = np.array(t)
    elif isinstance(t, np.ndarray) and t.ndim == 1:
        arr_t = t
    else:
        raise TypeError('Argument t must be a list or a numpy vector.')
    if Fmax <= 0:
        raise ValueError('Argument Fmax must be strictly positive.')

    y = np.zeros((x.size, t.size))
    for j in range(t.size):
        F = arr_t[j] * Fmax
        for i in range(x.size):
            x_i = arr_x[i]
            if x_i < W[0]:
                y[i, j] = F * (x_i ** 2) * (3 * W[0] - x_i) / (6 * U[0] * U[1])
            else:
                y[i, j] = F * (W[0] ** 2) * (3 * x_i - W[0]) / (6 * U[0] * U[1])

    return y


class Surrogate:
    """
    Surrogate model for a probabilistic model parametrized by control parameters. Can be used to generate samples,
    or estimate lower and upper confidence bounds, for a given value of time (or pseudo-time e.g. frequency)
    and for given values of the control parameters.
    """
    def __init__(self, data, n_Y, vec_t):
        self.data = data
        self.n_Y = n_Y
        self.vec_t = vec_t

        self.idx_t = None
        self.W = None
        self.surrogate_gkde = None
        self.lower_confidence_bound = None
        self.upper_confidence_bound = None

    def compute_surrogate_gkde(self, idx_t):
        """"""
        self.idx_t = idx_t
        self.surrogate_gkde = gkde(self.data[:, idx_t, :], bw_method='silverman')

    def sample(self, n_samples):
        """"""
        samples = self.surrogate_gkde.resample(n_samples)

        return samples

    def conditional_sample(self, W, n_samples):
        """"""
        samples = self.surrogate_gkde.conditional_resample(n_samples,
                                                           x_cond=W[:, np.newaxis].transpose(),
                                                           dims_cond=list(range(self.n_Y, self.data.shape[0])))
        samples = np.squeeze(samples, axis=-1)
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

    def compute_conditional_cdf_component(self, y, idx_y, W):
        """"""
        y_data = self.data[idx_y, self.idx_t, :]
        y_mean = np.mean(y_data)
        centered_y_data = y_data - np.ones(y_data.shape) * y_mean
        y_std = np.sqrt(np.dot(centered_y_data, centered_y_data.T) / (y_data.shape[0] - 1))
        if y_std <= 1e-6:
            y_std = 1.
        centered_reduced_y_data = centered_y_data / y_std

        centered_reduced_y = (y - y_mean) / y_std

        W_data = self.data[self.n_Y:, self.idx_t, :]
        W_mean = np.mean(W_data, axis=1)
        centered_W_data = W_data - np.tile(W_mean[:, np.newaxis], (1, W_data.shape[1]))
        W_covar = np.dot(centered_W_data, centered_W_data.T) / (W_data.shape[1] - 1)
        W_stds = np.sqrt(np.diag(W_covar))
        W_stds[W_stds <= 1e-6] = 1.
        centered_reduced_W_data = np.divide(centered_W_data, np.tile(W_stds[:, np.newaxis], (1, W_data.shape[1])))

        centered_reduced_W = np.divide(W - W_mean, W_stds)

        s = np.power(4 / (self.data.shape[1] * (2 + self.vec_t.size + W_data.shape[0])),
                     1 / (self.vec_t.size + W_data.shape[0] + 4))
        numerator = 0.
        denominator = 0.
        for i in range(self.data.shape[1]):
            w_term = np.exp(-(np.linalg.norm(centered_reduced_W - centered_reduced_W_data[:, i]) ** 2) / (2 * s ** 2))
            F_tilde = 0.5 * (1 + erf((centered_reduced_y - centered_reduced_y_data[i]) / (s * np.sqrt(2))))
            numerator += F_tilde * w_term
            denominator += w_term

        CDF_value = numerator / denominator

        return CDF_value

    # def compute_conditional_confidence_interval_old(self, W, p_confidence=0.95,
    #                                                 Y_lower_bound=-1e2, Y_upper_bound=1e2, n_points=10000):
    #     """"""
    #     arr_y = np.linspace(Y_lower_bound, Y_upper_bound, n_points)
    #     Y_lower_confidence_bound = np.zeros((self.n_Y, ))
    #     Y_upper_confidence_bound = np.zeros((self.n_Y, ))
    #     for i in range(self.n_Y):
    #         CDF_values = np.zeros((n_points, ))
    #         for j in range(n_points):
    #             CDF_values[j] = self.compute_conditional_cdf_component(arr_y[j], i, W)
    #         idx_lower_confidence_bound = (np.abs(CDF_values - (1 - p_confidence))).argmin()
    #         idx_upper_confidence_bound = (np.abs(CDF_values - p_confidence)).argmin()
    #         Y_lower_confidence_bound[i] = arr_y[idx_lower_confidence_bound]
    #         Y_upper_confidence_bound[i] = arr_y[idx_upper_confidence_bound]
    #
    #     return Y_lower_confidence_bound, Y_upper_confidence_bound

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