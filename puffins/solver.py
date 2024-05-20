import numpy as np

def design_matrix(x, period, n_harmonics):
    omegas = 2. * np.pi * np.arange(1, n_harmonics + 1) / period
    X = np.zeros((len(x), 2 * n_harmonics))
    X[:,::2] = np.cos(omegas[None, :] * x[:, None]) # I'm dyin heah ## original hogg quote, will not be removing.
    X[:,1::2] = np.sin(omegas[None, :] * x[:, None])
    return X


def solve_ridgeRegression(x, y, period, K, alpha):
    X = design_matrix(x, period, K)
    wI = np.eye(X.shape[1])
    wI[0,0] = 0
    betas_w = np.linalg.solve( alpha*wI + X.T@X, X.T @ y)
    return betas_w

def solve_simple(x, y, period, K):

    X = design_matrix(x, period, K)
    betas = np.linalg.solve(X.T @ X, X.T @ y)
    # amps, resids, rank, S = np.linalg.lstsq(X, flux, rcond=None)

    return betas

def predict_simple(x, y, period, K):
    X = design_matrix(x, period, K)
    betas = solve_simple(x, y, period, K)
    reconstructed_y = X @ betas
    return X, betas, reconstructed_y

def predict_ridge(x, y, period, K, alpha=10):
    X = design_matrix(x, period, K)
    betas = solve_ridgeRegression(x, y, period, K, alpha)
    reconstructed_y = X @ betas
    return X, betas, reconstructed_y

