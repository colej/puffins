"""
Tuner module for hyperparameter optimization.

Implements a Tuner class that wraps hyperparameter
tuning functionality, calling out to functions in
`tuner_functions.py`.
"""

import optuna
import numpy as np

from typing import Any, Dict, Optional, Tuple
from sklearn.model_selection import KFold, cross_val_score


class Tuner:
    def __init__(self, LinearModel: any, hyperpars: any = None, 
                 n_trials: int = 50, direction: str = 'minimize',
                ) -> None:
        self.model = LinearModel
        self.hyperpars = hyperpars
        self.n_trials = n_trials
        self.direction = direction
        self.best_hyperpars = None
        self.best_score = None

    def run_tune(self,
        predictors: np.ndarray,
        targets: np.ndarray,
        ) -> None:
        """
        Run hyperparameter tuning on the given model and dataset.

        This method calls out to the helper function in `tuner_functions.py`.

        Parameters
        ----------
        predictors : np.ndarray
            Features of the training set.
        targets : np.ndarray
            Targets/labels of the training set.
        """
        self.best_hyperpars, self.best_score = simple_tune(
            model=self.model,
            predictors=predictors,
            targets=targets,
            hyperparams=self.hyperpars,
            n_trials=self.n_trials,
            direction=self.direction
        )
 
    def __repr__(self):
        if self.best_hyperpars is not None:
            summary_str =  "\n".join([f"{par}: {self.best_hyperpars[par]}" for par in self.best_hyperpars ])
        else:
            summary_str = "No hyperparameters tuned yet."
        return f"Tuner:\n {summary_str}"



# class Objective(object):
#     def __init__(self, predictors, targets, solver, **kwargs):
#         self.predictors = predictors
#         self.targets = targets
#         self.solver = solver

#         pass

## Functions


def simple_tune( model: Any,
    predictors: np.array,
    targets: np.array,
    hyperparams: Dict[str, Any],
    n_trials: int,
    direction: str,
    **kwargs
) -> Tuple[Dict[str, Any], float]:
    """
    Tune a model's hyperparameters using Optuna and K-Fold cross-validation.

    Parameters
    ----------
    model : Any
        The model (e.g., sklearn estimator) to be tuned.
    predcitors : np.ndarray
        Features of the training set.
    targets : Any
        Targets/labels of the training set.
    hyperparams : dict
        Dictionary specifying hyperparameter distributions or discrete values.
        Example of a log-uniform distribution:
            { "feature_width": (1e-5, 1.0, "log") }
        Example of a uniform distribution:
            { "period": (0.0, 1.0, "uniform") }
    n_trials : int
        Number of trials for the Optuna study.
    direction : str
        "maximize" or "minimize" the objective metric.

    Returns
    -------
    best_params : dict
        The best hyperparameters found by Optuna.
    best_score : float
        The best score from tuning run.
    """

    def objective(trial: optuna.Trial, model: any, predictors: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # Construct the parameter set for each trial
        trial_params = {}
        for param_name, dist in hyperparams.items():
            # If user indicated a log-uniform distribution
            if len(dist) == 3 and dist[2] == "log":
                low, high, _ = dist
                trial_params[param_name] = trial.suggest_loguniform(param_name, low, high)
            elif param_name == "n_harmonics":
                low, high = dist[0], dist[1]
                trial_params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                # Default to uniform distribution
                low, high = dist[0], dist[1]
                trial_params[param_name] = trial.suggest_float(param_name, low, high)

        suggested_values = {key: trial_params[key] for key in hyperparams.keys()}

        # Update model parameters
        if 'feature_weighting_width' in suggested_values:
            model.feature_weighting_width = trial_params['feature_weighting_width']
        model.set_X_kwargs(update=True, **suggested_values)
        model.set_solver_kwargs(update=True, **suggested_values)
        model.set_X_train(predictors)
        
        model.train(targets)

        return model.trained_mse


    study = optuna.create_study(direction=direction)
    study.optimize( lambda trial:  objective(trial, model, predictors, targets, **kwargs), n_trials=n_trials)

    return study.best_params, study.best_value