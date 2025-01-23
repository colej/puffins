import inspect
import numpy as np

from sklearn.model_selection import KFold
from .solver import solve, get_solvers
from .basis_functions import basis_constant, basis_linear, basis_quadratic
from .utils import construct_design_matrix


class TimeSeries:
    def __init__(self, x, y, epsilon=None):
        """
        Initialize the time series object.

        Parameters:
        - times: np.ndarray
            Array of time points.
        - observations: np.ndarray
            Array of observed values.
        - uncertainties: np.ndarray
            Array of uncertainties corresponding to observations.
        """
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self.ph = None
        self.models = {}
        self.residuals = {}
        self.summary = None
        self._compute_summary()


    def compute_phase(self, period, t0=0):
        self.ph = ((self.x - t0) / period) % 1

    def _compute_summary(self):
        """Compute summary statistics for the time series."""
        summary = {
            "Time base of observations": self.x.max() - self.x.min(),
            "Number of data points": len(self.x),
            "Median time step": np.median(np.diff(np.sort(self.x))),
            "Number of models computed": len(self.models),
                  }
        self.summary = summary
        

    def __repr__(self):
        """String representation for debugging."""
        summary_str = "\n".join([f"{stat}: {val}" for stat, val in self.summary.items()])
        return f"TimeSeries with properties:\n{summary_str}"

    def add_model(self, model):
        self.models[model.method] = model
        self.residuals[model.method] = self.y - model.predict(self.x)[1]
        self._compute_summary()



class LinearModel:
    def __init__(self, time_series, method='wls', basis_functions=None, feature_embedding=None, 
                 feature_weighting_function=None, feature_width=None, **kwargs):
        """
        Initialize the model object.

        Parameters:
        - time_series: TimeSeries
            An instance of the TimeSeries class.
        - basis_functions: list of callables
            List of functions to transform time points into features.
        """
        self.time_series = time_series
        self.method = method
        self.basis_functions = basis_functions
        self.feature_embedding = feature_embedding
        self.feature_weights = None
        self.feature_weighting_function = feature_weighting_function
        self.feature_weighting_width = feature_width
        self.X_train = None
        self.coefficients = None
        self.trained_model = None
        self.summary = None
        self.solver = get_solvers(self.method)

        # Extract keyword arguments for construct_design_matrix
        X_list = list(inspect.signature(construct_design_matrix).parameters)
        self.X_kwargs = {key: kwargs[key] for key in X_list if key in kwargs}

        # # Extract keyword arguments for solve
        solver_list = list(inspect.signature(self.solver).parameters)
        self.solver_kwargs = {key: kwargs[key] for key in solver_list if key in kwargs}
        # self.kwargs = kwargs

        self._compute_summary()


    def _compute_summary(self):
        """Compute summary statistics for the time series."""
        summary = {
            "Basis functions": [f"{inspect.getsource(bf)}" for bf in self.basis_functions],
            "Feature embedding": len(self.feature_embedding),
            "Regression method": self.method,
            "Regression kwargs": self.solver_kwargs,
            "Design matrix kwargs": self.X_kwargs
                  }
        self.summary = summary


    def __repr__(self) -> str:
        summary_str = "\n".join([f"{stat}: {val}" for stat, val in self.summary.items() ])
        return f"Linear Model with properties: \n {summary_str}"


    def set_X_train(self):
            self.X_train, weighting_input = construct_design_matrix(x=self.time_series.x, basis_functions=self.basis_functions, feature_embedding=self.feature_embedding, **self.X_kwargs)
            if self.feature_weighting_function:
                self.feature_weights = self.feature_weighting_function(weighting_input, self.feature_weighting_width)
                self.solver_kwargs['L'] = self.feature_weights
            else:
                self.feature_weights = None

    def train(self):
        """
        Fit the model to the time series data.

        Parameters:
        - method: str, optional
            Method to use for fitting (default: "least_squares").
        """

        if self.X_train is None:
            raise ValueError("Design matrix has not been constructed. Call construct_design_matrix first.")


        self.coefficients = self.solver(self.X_train, self.time_series.y, **self.solver_kwargs)
        self.trained_model = self.X_train @ self.coefficients


    def predict(self, xpredict):
        """
        Predict values at new time points.

        Parameters:
        - times: np.ndarray
            Array of new time points.

        Returns:
        - predictions: np.ndarray
            Predicted values.
        """

        X_predict, _ = construct_design_matrix(xpredict, self.basis_functions, self.feature_embedding, **self.X_kwargs)

        return X_predict, X_predict @ self.coefficients



    # def cross_validate(self, n_splits=5):
    #     """
    #     Perform K-Fold cross-validation.

    #     Parameters:
    #     - n_splits: int, optional
    #         Number of splits for K-Fold (default: 5).

    #     Returns:
    #     - errors: list
    #         List of validation errors for each fold.
    #     """
    #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    #     errors = []

    #     for train_idx, val_idx in kf.split(self.time_series.times):
    #         train_times = self.time_series.times[train_idx]
    #         train_obs = self.time_series.observations[train_idx]
    #         train_unc = self.time_series.uncertainties[train_idx]

    #         val_times = self.time_series.times[val_idx]
    #         val_obs = self.time_series.observations[val_idx]

    #         temp_series = TimeSeries(train_times, train_obs, train_unc)
    #         temp_model = Model(temp_series, self.basis_functions)
    #         temp_model.construct_design_matrix()
    #         temp_model.fit()

    #         predictions = temp_model.predict(val_times)
    #         error = np.mean((predictions - val_obs)**2)
    #         errors.append(error)

    #     return errors

    # def bootstrap_uncertainties(self, n_bootstrap=1000):
    #     """
    #     Estimate uncertainties on coefficients using bootstrapping.

    #     Parameters:
    #     - n_bootstrap: int, optional
    #         Number of bootstrap resamples (default: 1000).

    #     Returns:
    #     - uncertainties: np.ndarray
    #         Bootstrap standard deviations for coefficients.
    #     """
    #     coefficients_samples = []

    #     for _ in range(n_bootstrap):
    #         indices = np.random.choice(len(self.time_series.times), len(self.time_series.times), replace=True)
    #         boot_times = self.time_series.times[indices]
    #         boot_obs = self.time_series.observations[indices]
    #         boot_unc = self.time_series.uncertainties[indices]

    #         boot_series = TimeSeries(boot_times, boot_obs, boot_unc)
    #         boot_model = Model(boot_series, self.basis_functions)
    #         boot_model.construct_design_matrix()
    #         boot_model.fit()
    #         coefficients_samples.append(boot_model.coefficients)

    #     return np.std(coefficients_samples, axis=0)



# Example Usage
# time_series = TimeSeries(times=[0, 1, 2, 3], observations=[1, 2, 1.5, 3], uncertainties=[0.1, 0.2, 0.1, 0.3])
# model = Model(time_series, basis_functions=[basis_linear, basis_quadratic, basis_sine])
# model.construct_design_matrix()
# model.fit()
# predictions = model.predict([0.5, 1.5, 2.5])
