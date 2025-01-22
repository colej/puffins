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
    design_matrix : np.ndarray
        The design matrix
    feature_weights : dict
        Additional outputs specific to the feature embedding, if applicable
    """

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
            if basis == basis_polynomial:
                design_matrix[i:,] = basis(x, kwargs.get("degree", 1))
            else:
                design_matrix[i:,] = basis(x)

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
            
    if not basis_functions and not feature_embedding:
        raise ValueError("Either basis_functions or feature_embedding must be provided.")

    return design_matrix,  feature_weights


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
