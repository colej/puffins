import numpy as np

np.random.seed(18443280306) # The General Insurance phone number to enforce reproducibility
RCOND = 1e-14 # condition number below which `np.linalg.lstsq()` zeros out eigen/singular values



def solve_ridgeRegression(X, y, alpha, weights=None):

    if weights is None:
        weights = np.ones(X.shape[0])

    XTCinv = X.T / weights
    XTCinvX = XTCinv @ X
    XTCinvY = XTCinv @ y
        
    idd = np.diag_indices_from(XTCinvX)
    XTCinvX[idd] += alpha
    
    betas = np.linalg.solve(XTCinvX, XTCinvY)
    return betas


def solve_fwols(X, omegas, y, width=None, weighted=False, weights=None):

    n, p = X.shape

    if width is None:
        width = 0.1 / (2.*np.pi)
    else:
        width /= (2. * np.pi)

    
    ## Assumes the approximation of a periodic version of the Matern 3/2 kernal
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
    A[idd] += ( (width**2) * (omegas**2) + 1 )
    betas = np.linalg.solve( A, B)
    return betas


def solve_wls(xs, ys, ws=None):
    n, p = xs.shape
    if ws is None:
        ws = np.ones(n)
    return np.linalg.lstsq(ws[:, None] * xs , ws * ys, rcond=RCOND)[0].T

def solve_ols(X, y, weights=None):

    if weights is not None:
        Cinv = np.linalg.inv(np.diag(weights))
        betas = np.linalg.solve(X.T@Cinv@X, X.T@Cinv@y)
    else:
        betas = np.linalg.solve(X.T @ X, X.T @ y)

    return betas




def train_feature_weighted_ols(xs, ys, ws=None):
    n, p = xs.shape
    if ws is None:
        ws = np.ones(p)
    return np.linalg.lstsq(xs * ws[None, :], ys, rcond=RCOND)[0].T * ws



# def solve_ridgeRegression(x, y, period, K, alpha, weighted=False, weights=None):
#     X,_ = design_matrix(x, period, K)
# #     wI = np.eye(X.shape[1])
# #     wI[0,0] = 0
# #     betas = np.linalg.solve( alpha*wI + X.T@X, X.T @ y)

#     if weighted:
#         Cinv = np.linalg.inv(np.diag(weights))
#         A = X.T@Cinv@X
#         B = X.T@Cinv@y
#     else:
#         A = X.T@X
#         B = X.T@y
        
#     idd = np.diag_indices_from(A)
#     A[idd] += alpha
    
#     betas = np.linalg.solve(A, B)
#     return betas


# def solve_approxGP(x, y, period, K, alpha, width=None, weighted=False, weights=None):

#     if width is None:
#         width = 0.1*period / (2.*np.pi)
#     else:
#         width /= (2. * np.pi)

    
#     ## Assumes the approximation of a periodic version of the Matern 3/2 kernal
#     X,diag_omegas = design_matrix(x, period, K)
    
#     # Implement eq. 23 from https://arxiv.org/pdf/2101.07256
#     # In this case, we're going to use the same X for our predictions
#     # however, we could propose a new X* where we want to predict
#     # the regression at times t*
#     if weighted:
#         Cinv = np.linalg.inv(np.diag(weights))
#         A = X.T@Cinv@X
#         B = X.T@Cinv@y
#     else:
#         A = X.T@X
#         B = X.T@y
    
#     idd = np.diag_indices_from(A)
#     A[idd] += alpha * ( (width**2) * (diag_omegas**2) + 1 )
#     betas = np.linalg.solve( A, B)
#     return betas

# def solve_simple(x, y, period, K, weighted=False, weights=None):

#     X,_ = design_matrix(x, period, K)
#     betas = np.linalg.solve(X.T @ X, X.T @ y)
#     # amps, resids, rank, S = np.linalg.lstsq(X, flux, rcond=None)

#     return betas


# def predict_simple(x, y, period, K):
#     X,_ = design_matrix(x, period, K)
#     betas = solve_simple(x, y, period, K)
#     reconstructed_y = X @ betas
#     return X, betas, reconstructed_y


# def predict_ridge(x, y, period, K, alpha=1):
#     X,_ = design_matrix(x, period, K)
#     betas = solve_ridgeRegression(x, y, period, K, alpha)
#     reconstructed_y = X @ betas
#     return X, betas, reconstructed_y


# def predict_approxGP(x, y, period, K, alpha=1, width=None):
#     X,_ = design_matrix(x, period, K)
#     betas = solve_ridgeRegression(x, y, period, K, alpha, width)
#     reconstructed_y = X @ betas
#     return X, betas, reconstructed_y
