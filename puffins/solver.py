import numpy as np


def design_matrix(x, period, n_harmonics):

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


def solve_ridgeRegression(x, y, period, K, alpha, weighted=False, weights=None):
    X,_ = design_matrix(x, period, K)
#     wI = np.eye(X.shape[1])
#     wI[0,0] = 0
#     betas = np.linalg.solve( alpha*wI + X.T@X, X.T @ y)

    if weighted:
        Cinv = np.linalg.inv(np.diag(weights))
        A = X.T@Cinv@X
        B = X.T@Cinv@y
    else:
        A = X.T@X
        B = X.T@y
        
    idd = np.diag_indices_from(A)
    A[idd] += alpha
    
    betas = np.linalg.solve(A, B)
    return betas


def solve_approxGP(x, y, period, K, alpha, width=None, weighted=False, weights=None):

    if width is None:
        width = 0.1*period / (2.*np.pi)
    else:
        width /= (2. * np.pi)

    
    ## Assumes the approximation of a periodic version of the Matern 3/2 kernal
    X,diag_omegas = design_matrix(x, period, K)
    
    # Implement eq. 23 from https://arxiv.org/pdf/2101.07256
    # In this case, we're going to use the same X for our predictions
    # however, we could propose a new X* where we want to predict
    # the regression at times t*
    if weighted:
        Cinv = np.linalg.inv(np.diag(weights))
        A = X.T@Cinv@X
        B = X.T@Cinv@y
    else:
        A = X.T@X
        B = X.T@y
    
    idd = np.diag_indices_from(A)
    A[idd] += alpha * ( (width**2) * (diag_omegas**2) + 1 )
    betas = np.linalg.solve( A, B)
    return betas

def solve_simple(x, y, period, K, weighted=False, weights=None):

    X,_ = design_matrix(x, period, K)
    betas = np.linalg.solve(X.T @ X, X.T @ y)
    # amps, resids, rank, S = np.linalg.lstsq(X, flux, rcond=None)

    return betas


def predict_simple(x, y, period, K):
    X,_ = design_matrix(x, period, K)
    betas = solve_simple(x, y, period, K)
    reconstructed_y = X @ betas
    return X, betas, reconstructed_y


def predict_ridge(x, y, period, K, alpha=1):
    X,_ = design_matrix(x, period, K)
    betas = solve_ridgeRegression(x, y, period, K, alpha)
    reconstructed_y = X @ betas
    return X, betas, reconstructed_y

def predict_approxGP(x, y, period, K, alpha=1, width=None):
    X,_ = design_matrix(x, period, K)
    betas = solve_ridgeRegression(x, y, period, K, alpha, width)
    reconstructed_y = X @ betas
    return X, betas, reconstructed_y
