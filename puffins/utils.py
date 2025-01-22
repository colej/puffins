import numpy as np

from .basis_functions import *


def construct_design_matrix(x, basis_functions=None, feature_embedding=None, **kwargs):
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
    X : np.ndarray
        The design matrix
    additional_outputs : dict
        Additional outputs specific to the feature embedding, if applicable
    """
    design_matrix_components = []
    feature_weights = []
    if basis_functions:
        n_bases = len(basis_functions)
    else:
        n_bases = 0
    if feature_embedding:
        if feature_embedding == 'fourier':
            n_embeddings = 2 * kwargs.get('n_harmonics', 1)
        elif feature_embedding == 'spherical_harmonics':
            n_embeddings = kwargs.get('n_harmonics', 1)
    else:
        n_embeddings = 0
  
    n_features = n_bases + n_embeddings
    design_matrix = np.ones((len(x), n_features))
    feature_weights = np.zeros(n_features)
  
    # Include basis functions if provided
    if basis_functions is not None:
        for i,basis in enumerate(basis_functions):
            design_matrix[i:,].append(basis(x))

    # Include feature embedding if specified
    if feature_embedding:
        if feature_embedding == "fourier":
            period = kwargs.get("period", 1.0)
            n_harmonics = kwargs.get("n_harmonics", 1)
            omegas = 2. * np.pi * np.arange(1, n_harmonics + 1) / period

            design_matrix[:,n_bases::2] = basis_cosine(x,omegas)
            feature_weights[n_bases::2] = omegas
            design_matrix[:,n_bases+1::2] = basis_sine(x,omegas)
            feature_weights[n_bases+1::2] = omegas
            
    if not design_matrix_components:
        raise ValueError("Either basis_functions or feature_embedding must be provided.")

    # Combine all components into the design matrix
    # X = np.column_stack(design_matrix_components)
    return design_matrix,  feature_weights



def construct_design_matrix_legacy(x, period, n_harmonics):

    """
    Parameters
    ----------
    x : np.ndarray
        The times of the observations
    period : float
        The period of the signal that we're trying to model
    n_harmonics : int
        The number of harmonics to include in the design matrix
    
    Returns
    -------
    X : np.ndarray
        The design matrix
    omega_list : np.ndarray
        The list of frequencies that the design matrix is sensitive to
    """

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



def sort_on_x(x,y,yerr):
    zipp = list(zip(x,y,yerr))
    zipp.sort(key=lambda x:x[0])
    x,y, yerr = list(zip(*zipp))
    return x, y, yerr


def cross_validate_hyperparams(x, y, yerr, period, K, n_folds=5, n_trials=50):

    # Create an Optuna study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, x, y, yerr, period, K, n_folds=n_folds), n_trials=n_trials)

    # Display the best alpha and corresponding score
    print(f"Best width: {study.best_params['width']}")
    print(f"Best MSE: {study.best_value}")

    return study.best_params['width']


def objective(trial, x, y, yerr, period, K, n_folds=5):

    width = trial.suggest_loguniform("width", 1e-3, 100.)  # Log-uniform search space
    
    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mse_scores = []
    

    for train_index, test_index in kf.split(x):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if yerr is None:
            w_train = None
        else:
            w_train, _ = 1./yerr[train_index]**2, 1./yerr[test_index]**2

        # Get the design matrices for the training and test sets
        X_train, omegas_train = design_matrix(x_train, period, K)
        X_test, _ = design_matrix(x_test, period, K)


        # Call your custom Ridge regression function
        betas_ = solve_approxGP(
            X_train, 
            omegas_train,
            y=y_train, 
            width=width,
            weights=w_train
        )
        
        # Make predictions on the test set
        y_pred = X_test @ betas_
        
        # Compute mean squared error
        mse = np.mean((y_test - y_pred) ** 2)
        mse_scores.append(mse)
    
    # Return the average MSE across folds
    return np.mean(mse_scores)
