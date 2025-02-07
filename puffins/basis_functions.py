import numpy as np

def basis_constant(
                    x: np.ndarray
                   ) -> np.ndarray:
    """
    Constant basis function.
    Returns an array of ones with the same shape as x.

    Parameters:
    - x: np.ndarray
        Array of input values

    Returns:
    - np.ndarray
        Array of ones with the same shape as x
    """

    return np.ones_like(x)


def basis_linear(
                 x: np.ndarray
                 ) -> np.ndarray:
    """
    Linear basis function.
    Returns x.
    
    Parameters:
    - x: np.ndarray
        Array of input values
        
    Returns:
    - np.ndarray
        Array of input values
    """

    return x


def basis_quadratic(
                    x: np.ndarray
                    ) -> np.ndarray:

    """
    Quadratic basis function.
    Returns x^2.

    Parameters:
    - x: np.ndarray
        Array of input values

    Returns:
    - np.ndarray
        Array of input values squared
    """

    return x**2


def basis_polynomial(
                      x: np.ndarray, 
                      degree: int
                     ) -> np.ndarray :

    """
    Polynomial basis function.
    Returns x^degree.

    Parameters:
    - x: np.ndarray
        Array of input values

    - degree: int
        Degree of polynomial

    Returns:
    - np.ndarray
        Array of input values raised to the power of degree
    """

    return np.power(x,degree)


def basis_cosine(
                 x: np.ndarray, 
                 omega: np.ndarray
                ) -> np.ndarray :
    
    """
    Cosine basis function.
    Returns cos(omega*x).

    Parameters:
    - x: np.ndarray
        Array of input values
    
    - omega: np.ndarray
        Array of angular frequencies
    
    Returns:
    - np.ndarray
        Array of cos(omega*x)
    """

    return np.cos(omega[None,:] * x[:,None])


def basis_sine(
                x: np.ndarray, 
                omega: np.ndarray
               ) -> np.ndarray :

    """
    Sine basis function.
    Returns sin(omega*x).

    Parameters:
    - x: np.ndarray
        Array of input values
    
    - omega: np.ndarray
        Array of angular frequencies

    Returns:
    - np.ndarray
        Array of sin(omega*x)    
    """
    return np.sin(omega[None,:] * x[:,None])


# TODO:
# - Implement spherical harmonic basis functions
