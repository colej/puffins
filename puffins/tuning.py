import numpy as np
import optuna
from sklearn.model_selection import KFold


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