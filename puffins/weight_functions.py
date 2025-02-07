import numpy as np

def exponential(
                omegas: np.ndarray, 
                width: float
                ) -> np.ndarray :
    """
    Exponential weight function.
    Same as Matern 1/2 kernel.

    Parameters
    ----------
    - omegas : np.ndarray
        The frequencies
    - width : float
        The width of the kernel
    
    Returns
    -------
    - np.ndarray
        The kernel values
    

    NOTES:
    - Originally written by D.W. Hogg
    - The "width" is in spatial separation, so it is really an inverse width here
    in Fourier space.

    """

    return np.sqrt(np.sqrt(2. / np.pi) / (1. / width + width * omegas ** 2))


def exponential_squared(
                        omegas: np.ndarray, 
                        width: float
                        ) -> np.ndarray :
  """
  Exponential squared weight function.

  Parameters
  ----------
  - omegas : np.ndarray
      The frequencies
  - width : float
      The width of the kernel
  
  Returns
  -------
  - np.ndarray
      The kernel values
  

  NOTES:
  - Originally written by D.W. Hogg
  - The "width" is in spatial separation, so it is really an inverse width here
    in Fourier space.

  """
  return np.sqrt(np.exp(-0.5 * omegas ** 2 * width ** 2))


def matern32(
             omegas: np.ndarray, 
             width: np.ndarray
            ) -> np.ndarray :
    """
    Matern 3/2 kernel.

    Parameters
    ----------
    - omegas : np.ndarray
        The frequencies
    - width : float
        The width of the kernel

    Returns
    -------
    - np.ndarray
        The kernel values
    
    NOTES:
    - Originally written by D.W. Hogg
    - The "width" is in spatial separation, so it is really an inverse width here
    in Fourier space.

    """

    return 1. / (width ** 2 * omegas ** 2 + 1.)


def matern52(
             omegas: np.ndarray, 
             width: np.ndarray
            ) -> np.ndarray :
    """
    Matern 5/2 kernel.

    Parameters
    ----------
    - omegas : np.ndarray
        The frequencies
    - width : float
        The width of the kernel

    Returns
    -------
    - np.ndarray
        The kernel values
    
    NOTES:
    - Originally written by D.W. Hogg
    - The "width" is in spatial separation, so it is really an inverse width here
    in Fourier space.
    """

    return 1. / (width ** 2 * omegas ** 2 + 1.)**2