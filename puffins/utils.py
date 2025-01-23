import numpy as np

from .basis_functions import *


def construct_design_matrix(x, basis_functions=None, feature_embedding=None, 
                            period=None, n_harmonics=1, polynomial_degree=None, **kwargs):
    """
    Generate a design matrix based on provided basis functions and/or feature embeddings.

    Parameters
    ----------
    x : np.ndarray
        The times of the observations
    basis_functions : list of callables, optional
        List of functions to transform time points into features
    feature_embedding : str, optional
        Type of feature embedding to use (e.g., "fourier")
    kwargs : dict
        Additional parameters for the feature embedding (e.g., period, n_harmonics)

    Returns
    -------
    design_matrix : np.ndarray
        The design matrix
    feature_weights : dict
        Additional outputs specific to the feature embedding, if applicable
    """

    if basis_functions:
        n_bases = len(basis_functions)
    else:
        n_bases = 0
    if feature_embedding:
        if feature_embedding == 'fourier':
            if n_harmonics is None:
                raise ValueError("Fourier feature embedding requires n_harmonics.")
            n_embeddings = 2 * n_harmonics
        elif feature_embedding == 'spherical_harmonics':
            if n_harmonics is None:
                raise ValueError("Spherical harmonics feature embedding requires n_harmonics.")
            n_embeddings = n_harmonics
    else:
        n_embeddings = 0
  
    n_features = n_bases + n_embeddings
    design_matrix = np.ones((len(x), n_features))
    feature_weights = np.zeros(n_features)
  
    # Include basis functions if provided
    if basis_functions is not None:
        for i,basis in enumerate(basis_functions):
            if basis == basis_polynomial:
                if polynomial_degree is None:
                    raise ValueError("Polynomial basis function requires a degree.")
                design_matrix[:,i] = basis(x, polynomial_degree)
            else:
                design_matrix[:,i] = basis(x)

    # Include feature embedding if specified
    if feature_embedding:
        if feature_embedding == "fourier":
            if period is None:
                raise ValueError("Fourier feature embedding requires a period.")
            if n_harmonics is None:
                raise ValueError("Fourier feature embedding requires n_harmonics.")

            omegas = 2. * np.pi * np.arange(1, n_harmonics + 1) / period

            design_matrix[:,n_bases::2] = basis_cosine(x,omegas)
            feature_weights[n_bases::2] = omegas
            design_matrix[:,n_bases+1::2] = basis_sine(x,omegas)
            feature_weights[n_bases+1::2] = omegas
            
    if not basis_functions and not feature_embedding:
        raise ValueError("Either basis_functions or feature_embedding must be provided.")

    return design_matrix,  feature_weights



def sort_on_x(x,y,yerr):
    zipp = list(zip(x,y,yerr))
    zipp.sort(key=lambda x:x[0])
    x,y, yerr = list(zip(*zipp))
    return x, y, yerr
