from typing import Union
import numpy as np


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


def integrate_joint_pdf(joint_gkde, Y_mins, Y_maxs, W, n_points):
    """
    Integrates the joint pdf p_{Y, W}(y, w) with respect to y.

    Parameters
    ----------
    joint_gkde: scipy gaussian_kde object constructed using a dataset
    Y_mins: lower bounds of Y used for integration
    Y_maxs: upper bounds of Y used for integration
    W: vector of desired control parameters
    n_points: number of points used for the integration

    Returns
    -------
    value: result of the integration with respect to y, given control parameters W

    """


class Surrogate:
    """
    Surrogate model for a probabilistic model parametrized by control parameters. Can be used to generate samples,
    or estimate lower and upper confidence bounds, for a given value of time (or pseudo-time e.g. frequency)
    and for given values of the control parameters.
    """
    def __init__(self, dataset):
        self.dataset = dataset

        self.t = None
        self.joint_gkde = None
        self.W = None
        self.conditional_gkde = None

        self.surrogate_gkde = None
        self.lower_confidence_bound = None
        self.upper_confidence_bound = None

    def compute_joint_gkde(self, t):
        """"""
        self.t = t

    def compute_conditional_gkde(self, W):
        """"""
        self.W = W

    def compute_surrogate_gkde(self):
        """"""

    def compute_confidence_bounds(self, p_confidence=0.95):
        """"""

    def sample_surrogate(self, n_samples):
        """"""
        samples = self.surrogate_gkde.resample(n_samples)

        return samples