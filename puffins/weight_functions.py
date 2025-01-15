import numpy as np

def exponential(omegas, width):
    """
    Exponential weight function.
    Same as Matern 1/2 kernel.
    
    ## comments:
    - The "width" is in spatial separation, so it is really an inverse width here
    in Fourier space.

    """

    return np.sqrt(np.sqrt(2. / np.pi) / (1. / width + width * omegas ** 2))


def exponential_squared(omegas, width):
  """
  This function is chosen because we know the F.T. of its square.

  ## comments:
  - The "width" is in spatial separation, so it is really an inverse width here
    in Fourier space.

  ## bugs:
  - Must be synchronized with `kernel_function()`.
  """
  return np.sqrt(np.exp(-0.5 * omegas ** 2 * width ** 2))


def matern32(omegas, width):
    """
    This function is chosen because we know the F.T. of its square.

    ## comments:
    - The "width" is in spatial separation, so it is really an inverse width here
    in Fourier space.

    """

    return 1. / (width ** 2 * omegas ** 2 + 1.)


def matern52(omegas, width):
    """
    This function is chosen because we know the F.T. of its square.

    ## comments:
    - The "width" is in spatial separation, so it is really an inverse width here
    in Fourier space.

    """

    return 1. / (width ** 2 * omegas ** 2 + 1.)**2