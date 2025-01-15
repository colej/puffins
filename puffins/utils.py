import numpy as np


def get_design_matrix(x, period, n_harmonics):

    """
    Parameters
    ----------
    x : np.ndarray
        The times of the observations
    period : float
        The period of the signal that we're trying to model
    n_harmonics : int
        The number of harmonics to include in the design matrix
    
    Returns
    -------
    X : np.ndarray
        The design matrix
    omega_list : np.ndarray
        The list of frequencies that the design matrix is sensitive to
    """

    # We're going to do this as a fourier series, so our design matrix
    # necessarily needs to be the (1 cos() sin()) pairs for each
    # harmonic of the orbital period that we're considering

    # First, we're going to get all the possible cyclic frequencies
    # that we're concerned with, i.e. the N harmonics of the orbital
    # frequency
    omegas = 2. * np.pi * np.arange(1, n_harmonics + 1) / period

    # Next, we instantiate the design matrix to have 2 * n_harmonics + 1 columns
    # since we want an offset term and sin+cos term for each harmonic
    X = np.ones((len(x), 2 * n_harmonics + 1))

    # We do the same to keep track of the actual frequencies in case we
    # want to do the approximate GP regression
    omega_list = np.zeros(2 * n_harmonics + 1)
 
    # Consider the 0-frequency base term
    #X[:,0] = 1.
    omega_list[0] = 0.

    # Populate the cosine terms
    X[:,1::2] = np.cos(omegas[None, :] * x[:, None]) # I'm dyin heah ## original hogg quote, will not be removing.
    omega_list[1::2] = omegas

    # Populate the cosine terms
    X[:,2::2] = np.sin(omegas[None, :] * x[:, None])
    omega_list[2::2] = omegas

    return X, omega_list