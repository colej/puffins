import numpy as np

def basis_constant(x):
    return np.ones_like(x)


def basis_linear(x):
    return x


def basis_quadratic(x):
    return x**2


def basis_polynomial(x, degree):
    return np.power(x,degree)


def basis_cosine(x, omega):
    return np.cos(omega[None,:] * x[:,None])


def basis_sine(x, omega):
    return np.sin(omega[None,:] * x[:,None])


# TODO:
# - Implement spherical harmonic basis functions
