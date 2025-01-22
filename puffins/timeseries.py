import numpy as np

from sklearn.model_selection import KFold
from .solver import solve
from .basis_functions import basis_constant, basis_linear, basis_quadratic
from .utils import construct_design_matrix


class TimeSeries:
    def __init__(self, x, y, epsilon=None, period = None):
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
        self.period = period
        self.summary = self._compute_summary()
        self.models = {}
        self.residuals = {}

    def _compute_summary(self):
        """Compute summary statistics for the time series."""
        summary = {
            "Time base of observations": self.x.max() - self.x.min(),
            "Number of data points": len(self.x),
            "Median time step": np.median(np.diff(np.sort(self.x))),
            "Number of models computed": len(self.models),
                  }

        return summary

    def __repr__(self):
        """String representation for debugging."""
        summary_str = "\n".join([f"{stat}: {val}" for stat, val in self.summary.items()])
        return f"TimeSeries with properties:\n{summary_str}"



class Model:
    def __init__(self, time_series, method='wls', basis_functions=None, feature_embedding=None, **kwargs):
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
        self.kwargs = kwargs
        self.feature_weights = None
        self.design_matrix = None
        self.coefficients = None

    def call_design_matrix(self, train=True, **kwargs):
        if train:
            self.design_matrix = construct_design_matrix(self.time_series.x, self.basis_functions, self.feature_embedding, **self.kwargs)
        else:
            return construct_design_matrix(kwargs(), self.basis_functions, self.feature_embedding, **self.kwargs)
    
    def train(self, method="wls"):
        """
        Fit the model to the time series data.

        Parameters:
        - method: str, optional
            Method to use for fitting (default: "least_squares").
        """
        from solver import solve

        y = self.time_series.observations
        W = np.diag(1 / self.time_series.uncertainties**2)  # Weight matrix

        if self.design_matrix is None:
            raise ValueError("Design matrix has not been constructed. Call construct_design_matrix first.")

        self.coefficients = solve(self.design_matrix, self.time_series, W)

    def predict(self, xnew):
        """
        Predict values at new time points.

        Parameters:
        - times: np.ndarray
            Array of new time points.

        Returns:
        - predictions: np.ndarray
            Predicted values.
        """
        new_design_matrix = np.column_stack([basis(times) for basis in self.basis_functions])
        return new_design_matrix @ self.coefficients

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
