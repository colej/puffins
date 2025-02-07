import inspect
import numpy as np


from .solver import get_solvers
from .utils import construct_design_matrix

class DataSet:
    def __init__(self, predictors: np.ndarray , 
                 targets: np.ndarray , 
                 target_uncertainties: np.ndarray | None = None
                ):
        """
        Initialize the data object
        Parameters:
        - predictors: np.ndarray
            Array of predictors.
        - targets: np.ndarray
            Array of observed values.
        - epsilon: np.ndarray
            Array of uncertainties corresponding to predictors.
        """
        self.predictors = predictors
        self.targets = targets
        self.target_uncertainties = target_uncertainties
        self.regressors = {}
        self.trained_models = {}
        self.residuals = {}
        self.summary = None
        self._compute_summary()

    def _compute_summary(self) -> None:
        """Compute summary statistics for the time series."""
        summary = {
            "Number of data points": len(self.predictors),
            "Number of models computed": len(self.regressors),
                  }
        self.summary = summary
        

    def __repr__(self)  -> str:
        """String representation for debugging."""
        summary_str = "\n".join([f"{stat}: {val}" for stat, val in self.summary.items()])
        return f"TimeSeries with properties:\n{summary_str}"


    def add_model(self, 
                  model: any
                  ) -> None:
        self.regressors[model.method] = model
        self.trained_models[model.method] = model.predict(self.predictors)[1]
        self.residuals[model.method] = self.targets - self.trained_models[model.method]
        self._compute_summary()


class TimeSeries(DataSet):

    def __init__(self, 
                 predictors: np.array, 
                 targets: np.array, 
                 target_uncertainties: np.ndarray | None = None, 
                 period: float = 1., t0: float = 0.):
        super().__init__(predictors=predictors, targets=targets, target_uncertainties=target_uncertainties)
        self.period = period
        self.t0 = t0
        self.ph = None


    def compute_phase(self) -> None:
        self.ph = ((self.predictors - self.t0) / self.period) % 1


    def _compute_summary(self) -> None:
        """Compute summary statistics for the time series."""
        summary = {
            "Time base of dataset": self.predictors.max() - self.predictors.min(),
            "Median time-step of data set": np.median(np.diff(self.predictors)),
            "Number of data points": len(self.predictors),
            "Number of models computed": len(self.trained_models),
                  }
        self.summary = summary
        

    def __repr__(self) -> str:
        """String representation for debugging."""
        summary_str = "\n".join([f"{stat}: {val}" for stat, val in self.summary.items()])
        return f"TimeSeries with properties:\n{summary_str}"



class LinearModel:
    def __init__(self, 
                 method: str = 'wls', 
                 basis_functions: list or None = None, 
                 feature_embedding: str or None = None, 
                 feature_weighting_function: callable or None = None, 
                 feature_weighting_width: float or None = None, 
                 **kwargs):
        """
        Initialize the model object.

        Parameters:
        - DataSet: DataSet
            An instance of the DataSet class.
        - basis_functions: list of callables
            List of functions to transform predictors into features.
        """

        self.method = method
        self.basis_functions = basis_functions
        self.feature_embedding = feature_embedding
        self.feature_weighting_function = feature_weighting_function
        self.feature_weighting_width = feature_weighting_width
        self.feature_weight_inputs = None
        self.feature_weights = None
        self.X_train = None
        self.coefficients = None
        self.trained_model = None
        self.trained_mse = None
        self.summary = None
        self.solver = get_solvers(self.method)

        # Extract keyword arguments for construct_design_matrix
        # self.X_kwargs = {key: kwargs[key] for key in X_list if key in kwargs}
        # X_list = list(inspect.signature(construct_design_matrix).parameters)
        self.X_kwargs = {}
        self.set_X_kwargs(**kwargs)

        # # Extract keyword arguments for solve
        # solver_list = list(inspect.signature(self.solver).parameters)
        # self.solver_kwargs = {key: kwargs[key] for key in solver_list if key in kwargs}
        self.solver_kwargs = {}
        self.set_solver_kwargs(**kwargs)

        self._compute_summary()


    def _compute_summary(self) -> None:
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


    def set_X_kwargs(self, 
                     update: bool = False,
                     **kwargs
                     ) -> None:
        X_list = list(inspect.signature(construct_design_matrix).parameters)
        if update:
            for key in X_list:
                if key in kwargs:
                    self.X_kwargs[key] = kwargs[key]
        else:
            self.X_kwargs = {key: kwargs[key] for key in X_list if key in kwargs}



    def set_solver_kwargs(self,
                          update: bool = False, 
                          **kwargs
                          ) -> None:
        solver_list = list(inspect.signature(self.solver).parameters)
        self.solver_kwargs = {key: kwargs[key] for key in solver_list if key in kwargs}
        if update:
            for key in solver_list:
                if key in kwargs:
                    self.solver_kwargs[key] = kwargs[key]
            else:
                self.solver_kwargs = {key: kwargs[key] for key in solver_list if key in kwargs}


    def set_X_train(self, 
                    predictors: np.ndarray
                    ) -> None:
            self.X_train, self.feature_weight_inputs = construct_design_matrix(x=predictors, basis_functions=self.basis_functions, feature_embedding=self.feature_embedding, **self.X_kwargs)
            if self.feature_weighting_function:
                self.feature_weights = self.feature_weighting_function(self.feature_weight_inputs, self.feature_weighting_width)
                self.solver_kwargs['L'] = 1./(self.feature_weights)**2
            else:
                self.feature_weights = None


    def train(self, 
              targets: np.ndarray
              ) -> None:
        """
        Trains the model on the data.

        Parameters:
        - targets: np.ndarray
            Array of observed values
        """

        if self.X_train is None:
            raise ValueError("Design matrix has not been constructed. Call construct_design_matrix first.")

        self.coefficients = self.solver(self.X_train, targets, **self.solver_kwargs)
        self.trained_model = self.X_train @ self.coefficients
        self.trained_mse = np.mean((self.trained_model - targets)**2)



    def predict(
                self, 
                predictors: np.ndarray | None = None, 
                targets: np.ndarray | None = None,
                coefficients: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None ]:
        """
        Predict values at new data points.

        Parameters:
        - predictors: np.ndarray
            Array of new points to predict at

        - taregts (optional): np.ndarray
            Array of observed values to compute residuals.

        Returns:
        - model: np.ndarray
            Predicted values.
        """

        X_predict, _ = construct_design_matrix(x=predictors, basis_functions=self.basis_functions, 
                                            feature_embedding=self.feature_embedding, **self.X_kwargs)

        if coefficients is None:
            coefficients = self.coefficients

        fit = X_predict @ coefficients

        if targets is None:
            residuals = None
        else:
            residuals = targets - fit

        return X_predict, fit, residuals