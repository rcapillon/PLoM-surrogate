from typing import Union
import numpy as np


def model_sinc(W: np.ndarray, U: np.ndarray, t: Union[list, np.ndarray]):
    """
    Simple model generating a scalar time series from 2 control and 2 uncertain parameters.

    Parameters
    ----------
    W: 2-dimensional vector of control parameters
    U: 2-dimensional vector of uncertain parameters
    t: List of Nt time values for which the output is calculated

    Returns
    -------
    Y: Nt-dimensional vector of outputs (time series)

    """

    # Notes pour l'instant :
    # - loi normale centrée réduite pour U[0] ?
    # - loi beta shiftée (de 2pi ?) pour U[1] ?
    #   -> ou shiftée de 4 + U[0] pour introduire une dépendance entre les deux variables

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