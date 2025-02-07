import inspect
import numpy as np

np.random.seed(1234567) # The General Insurance phone number to enforce reproducibility
RCOND = 1e-14 # condition number below which `np.linalg.lstsq()` zeros out eigen/singular values


def solve_wls(
               X: np.ndarray, 
               y: np.ndarray, 
               W: np.ndarray | None = None
              ) -> np.ndarray :

    '''
    Weighted least squares solver.
    -------------------------------

    Parameters:
    - X: np.ndarray
        Design matrix

    - y: np.ndarray
        Target values

    - W: np.ndarray 
        Weights for the design matrix
        Assumes matrix with diagonal elements as 1/inverse variance on Y

    Returns:
    - beta_hat: np.ndarray
        Solved coefficients (betas)

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



def solve_generalRidge(
                        X: np.ndarray, 
                        y: np.ndarray, 
                        alpha: float = 0., 
                        W: np.ndarray | None = None
                       ) -> np.ndarray :
    '''
    Generalized least squares solver.
    -------------------------------

    Parameters:
    - X: np.ndarray
        Design matrix

    - y: np.ndarray
        Target values

    - W: np.ndarray
        Weights for the design matrix
        Assumes matrix with diagonal elements as 1/inverse variance on Y

    Returns:
    - beta_hat: np.ndarray 
        Solved coefficients (betas)

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


def solve_fw(
             X: np.ndarray, 
             y: np.ndarray, 
             W: np.ndarray | None = None, 
             L: np.ndarray | None = None
             ) -> np.ndarray :

    '''
    Feature weighted least squares solver.
    -------------------------------
    Parameters:
    - X: Design matrix
        np.ndarray of shape (n, p)
    
    - y: Target values
        np.ndarray of shape (n,)
    
    - L: Feature weights
        np.ndarray of shape (p,)
    
    - W: Weights for the targets
        np.ndarray of shape (n,)
        Assumes matrix with diagonal elements as 1/inverse variance on Y
    
    Returns:
    - beta_hat: Solved coefficients (betas)
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


def get_solvers( method: str ) -> callable:

    """
    Wrapper to get the solver function based on the method.

    Parameters:
    - method: str 
        Solver method. Options are 'ols', 'ridge', 'fwols'.

    Returns:
    - callable: 
        Solver function.

        """
    methods = {
        "wls": solve_wls,
        "ridge": solve_generalRidge,
        "fw": solve_fw,
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Supported methods: {list(methods.keys())}")
    return methods[method]


def solve(
          X: np.ndarray, 
          y: np.ndarray, 
          method: str = "wls", 
          **kwargs
          ) -> np.ndarray:
    """
    General solver wrapper.

    Parameters:
    - X: np.ndarray)
        Design matrix.
    - y: np.ndarray
        Target values.
    - method: str 
        Solver method. Options are 'ols', 'ridge', 'fwols'.
    - **kwargs 
        Additional arguments for the specific solver.

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