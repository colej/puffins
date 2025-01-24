import inspect
import numpy as np

np.random.seed(1234567) # The General Insurance phone number to enforce reproducibility
RCOND = 1e-14 # condition number below which `np.linalg.lstsq()` zeros out eigen/singular values


def solve_wls(X, y, W=None):

    '''
    Weighted least squares solver.
    -------------------------------

    Parameters:
    X: Design matrix

    y: Target values

    W: Weights for the design matrix
       Assumes matrix with diagonal elements as 1/inverse variance on Y

    Returns:
    beta_hat: Solved coefficients (betas)

    NOTES:
    - If W is not provided, it assumes OLS
    - If W is provided, it assumes GLS
    - If n > p, it assumes overdetermined and uses the normal equation
    - If n < p, it assumes underdetermined and uses the normal equation

    TODO: 
    - Implement the case where W is derived from a non-diagonal covariance matrix
    
    '''

    n, p = X.shape

    if W is None:
        W = np.ones(n)
    
    if n > p: # Overdetermined
        # beta_hat = (X.T * W * X)^-1 * X.T * W * y
        XTW = X.T * W
        return np.linalg.lstsq(XTW @ X, XTW @ y, rcond=RCOND)[0]

    else: # underdetermined
        # beta_hat = W * X.T (X @ W @ X.T)^-1 y
        return W * X.T @ np.linalg.lstsq( X * W @ X.T , y, rcond=RCOND)[0]



def solve_generalRidge(X, y, alpha=0., W=None):
    '''
    Generalized least squares solver.
    -------------------------------

    Parameters:
    X: Design matrix

    y: Target values

    W: Weights for the design matrix
       Assumes matrix with diagonal elements as 1/inverse variance on Y

    Returns:
    beta_hat: Solved coefficients (betas)

    NOTES:
    - If W is not provided, it assumes OLS
    - If W is provided, it assumes GLS
    - If n > p, it assumes overdetermined and uses the normal equation
    - If n < p, it assumes underdetermined and uses the normal equation

    TODO: 
    - Implement the case where W is derived from a non-diagonal covariance matrix

    '''

    n, p = X.shape

    if W is None:
        W = np.ones(n)
    
    XTW = X.T * W
    XTWX = XTW @ X

    XTWX[np.diag_indices_from(XTWX)] += alpha
    return np.linalg.lstsq(XTWX, XTW @ y, rcond=RCOND)[0]


def solve_fw(X, y, W=None, L=None):

    '''
    Feature weighted least squares solver.
    -------------------------------
    Parameters:
    X: Design matrix
        np.ndarray of shape (n, p)
    
    y: Target values
        np.ndarray of shape (n,)
    
    L: Feature weights
        np.ndarray of shape (p,)
    
    W: Weights for the targets
        np.ndarray of shape (n,)
        Assumes matrix with diagonal elements as 1/inverse variance on Y
    
    Returns:
    beta_hat: Solved coefficients (betas)
        np.ndarray of shape (p,)

    NOTES:
    - If weights is not provided, it assumes FWOLS
    - W = 1/C^2, where C is the uncertainty on the target values

    TODO: 
    - Implement the case where W is derived from a non-diagonal covariance matrix

    '''

    n, p = X.shape

    if W is None:
        weights = np.ones(n)
    if L is None:
        L = np.ones(p)


    ## Assumes the approximation of a periodic version of the Matern 3/2 kernal
    # Implement eq. 23 from https://arxiv.org/pdf/2101.07256
    # In this case, we're going to use the same X for our predictions
    # however, we could propose a new X* where we want to predict
    # the regression at times t*

    if n > p: # Overdetermined
        # Use eq. 23 from https://arxiv.org/pdf/2101.07256
        # beta_hat = (X.T * W * X + L)^-1 * X.T * W * y 
        # Identical to the generalized ridge regression! 
        # We just use special weights to approximate a Gaussian Process regression

        XTW = X.T * weights
        XTWX= XTW @ X
        XTWX[np.diag_indices_from(XTWX)] += L
        return np.linalg.lstsq(XTWX, XTW @ y, rcond=RCOND)[0]

    else: #  underdetermined
        # Use eq. 24 from https://arxiv.org/pdf/2101.07256
        # beta_hat = L^-1 @ X.T (X @ L^-1 @ X.T + 1/W)^-1 y

        LinvXT = (X / L).T
        XLinvXT = X @ LinvXT
        XLinvXT[np.diag_indices(n)] += 1./W
        return LinvXT @ np.linalg.lstsq(XLinvXT, y, rcond=RCOND)[0]


def get_solvers(method):
    methods = {
        "wls": solve_wls,
        "ridge": solve_generalRidge,
        "fw": solve_fw,
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Supported methods: {list(methods.keys())}")
    return methods[method]

def solve(X, y, method="wls", **kwargs):
    """
    General solver wrapper.

    Parameters:
    - X (np.ndarray): Design matrix.
    - y (np.ndarray): Target values.
    - method (str): Solver method. Options are 'ols', 'ridge', 'fwols'.
    - **kwargs: Additional arguments for the specific solver.

    Returns:
    - np.ndarray: Solved coefficients (betas).
    """
    solvers = {
        "wls": solve_wls,
        "ridge": solve_generalRidge,
        "fw": solve_fw,
    }

    if method not in solvers:
        raise ValueError(f"Unknown method '{method}'. Supported methods: {list(solvers.keys())}")

    solver = solvers[method]
    signature = list(inspect.signature(solver).parameters)
    solver_kwargs = {key: kwargs[key] for key in kwargs if key in signature}

    return solver(X, y, **solver_kwargs)



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
