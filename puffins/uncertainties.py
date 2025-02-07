import numpy as np
from tqdm import tqdm
from scipy import stats

class UncertaintyEstimator:
    def __init__( 
                  self, 
                  DataSet: any, 
                  Model: any, 
                  **kwargs
                 ) -> None:
        self.DataSet = DataSet
        self.Model = Model
        self.kwargs = kwargs
        self.loocv = None
        self.grouped_jk = None
        self.delete_d_jk = None
        self.mc = None
        self.mcmc = None


    def run_loocv_sampling( 
                            self, 
                            predictors: np.ndarray | None = None, 
                            targets: np.ndarray | None = None, 
                            **kwargs
                            ) -> dict:

        if predictors is None:
            predictors = self.DataSet.predictors
        if targets is None:
            targets = self.DataSet.targets

        self.loocv = LOOCV_sampling(predictors, targets, self.Model, **kwargs)
        return self.loocv

    def run_grouped_jackknife_sampling( self,
                                        predictors: np.ndarray | None = None, 
                                        targets: np.ndarray | None = None, 
                                        n_groups:int = 10,
                                        **kwargs
                                       ) -> dict:
        if predictors is None:
            predictors = self.DataSet.predictors
        if targets is None:
            targets = self.DataSet.targets

        self.grouped_jk =  grouped_jackknife_sampling( predictors, targets, self.Model, 
                                                       n_groups=n_groups, **kwargs)
        return self.grouped_jk


    def run_delete_d_jackknife_sampling( self,
                                        predictors: np.ndarray | None = None, 
                                        targets: np.ndarray | None = None, 
                                        n_groups:int = 10,
                                        n_delete: int = 50,
                                        **kwargs
                                       ) -> dict:
        if predictors is None:
            predictors = self.DataSet.predictors
        if targets is None:
            targets = self.DataSet.targets

        self.delete_d_jk =  delete_d_jackknife_sampling( predictors, targets, self.Model, 
                                                       n_groups=n_groups, n_delete=n_delete, **kwargs)
        return self.delete_d_jk


    def run_monte_carlo_sampling( self,
                         predictors: np.ndarray | None = None, 
                         targets: np.ndarray | None = None, 
                         target_uncertainties: np.ndarray | None = None, 
                         n_samples: int = 1000, 
                         **kwargs
                        ) -> dict:
        
        if predictors is None:
            predictors = self.DataSet.predictors

        if targets is None:
            targets = self.DataSet.targets

        if target_uncertainties is None:
            target_uncertainties = self.DataSet.target_uncertainties

        self.mc = monte_carlo_sampling( predictors=predictors, 
                              targets=targets, 
                              target_uncertainties=target_uncertainties,
                              Model=self.Model, 
                              n_samples=n_samples, 
                              **kwargs)
        return self.mc

    def run_mcmc_sampling(self):
        pass


def LOOCV_sampling(
           predictors: np.ndarray, 
           targets:np.ndarray, 
           Model: any, 
           **kwargs
          ) -> dict:
    """
    Leave-one-out cross-validation for regression coefficients.

    Parameters:
    - predictors (np.ndarray): predictor variables.
    - targets (np.ndarray): target values to train on / predict.
    - Model (class): Model class to use for fitting.
    - kwargs: Additional arguments for the solver.

    Returns:
    - coefs (np.ndarray): Coefficients for each fold.
    - coefs_mean (np.ndarray): Mean of coefficients.
    - coefs_std (np.ndarray): Standard deviation of coefficients.
    """
    n_samples = predictors.shape[0]
    coefs = []

    for i in tqdm(range(n_samples), desc="LOOCV (Delete-1 Jackkinfe Sampling)"):
        predictors_i = np.delete(predictors, i, axis=0)
        targets_i = np.delete(targets, i, axis=0)
        Model.set_X_train(predictors_i)
        Model.train(targets_i, **kwargs)
        coefs.append(Model.coefficients)

    coefs = np.array(coefs)
    var_ = get_delete_1_jk_var(coefs)
    return {'sampled_coefs':coefs, 
            'coefs_mean': np.mean(coefs, axis=0), 
            'coefs_var': var_}


def delete_d_jackknife_sampling(
                        predictors: np.ndarray, 
                        targets:np.ndarray, 
                        Model: any, 
                        n_groups:int = 10,
                        n_delete: int = 50, 
                        **kwargs
                        ) -> dict:
    """
    Grouped-Jackknife estimation of uncertainties for regression coefficients.

    Parameters:
    - predictors (np.ndarray): predictor variables.
    - targets (np.ndarray): target values to train on / predict.
    - Model (class): Model class to use for fitting.
    - d (int): Number of groups to split the data into.
    - kwargs: Additional arguments for the solver.

    Returns:
    - coefs (np.ndarray): Coefficients for each fold.
    - coefs_mean (np.ndarray): Mean of coefficients.
    - coefs_std (np.ndarray): Standard deviation of coefficients.

    """
    coefs = []

    n_samples = predictors.shape[0]

    for i in tqdm(range(n_groups), desc="Delete-d Jackkinfe Sampling"):
        idx = np.random.choice(n_samples, n_delete, replace=False)
        predictors_i = np.delete(predictors, idx, axis=0)
        targets_i = np.delete(targets, idx, axis=0)

        Model.set_X_train(predictors_i)
        Model.train(targets_i)

        coefs.append(Model.coefficients)


    coefs = np.array(coefs)
    var_ = get_delete_d_jk_var(n_samples, coefs, n_groups, n_delete)

    return {'sampled_coefs':coefs, 
            'coefs_mean': np.mean(coefs, axis=0),
            'coefs_var': var_}



def grouped_jackknife_sampling(
                        predictors: np.ndarray, 
                        targets:np.ndarray, 
                        Model: any, 
                        n_groups:int = 10, 
                        **kwargs
                        ) -> dict:
    """
    Grouped-Jackknife estimation of uncertainties for regression coefficients.

    Parameters:
    - predictors (np.ndarray): predictor variables.
    - targets (np.ndarray): target values to train on / predict.
    - Model (class): Model class to use for fitting.
    - d (int): Number of groups to split the data into.
    - kwargs: Additional arguments for the solver.

    Returns:
    - coefs (np.ndarray): Coefficients for each fold.
    - coefs_mean (np.ndarray): Mean of coefficients.
    - coefs_std (np.ndarray): Standard deviation of coefficients.

    """
    coefs = []

    n_samples = predictors.shape[0]
    n_delete = n_samples // n_groups
    for i in tqdm(range(n_groups), desc="Grouped Jackkinfe Sampling"):
        idx = np.random.choice(n_samples, n_delete, replace=False)
        predictors_i = np.delete(predictors, idx, axis=0)
        targets_i = np.delete(targets, idx, axis=0)

        Model.set_X_train(predictors_i)
        Model.train(targets_i)

        coefs.append(Model.coefficients)


    coefs = np.array(coefs)
    var_ = get_grouped_jk_var(coefs, n_groups)

    return {'sampled_coefs':coefs, 
            'coefs_mean': np.mean(coefs, axis=0),
            'coefs_var': var_}



def monte_carlo_sampling( 
                 predictors: np.ndarray, 
                 targets: np.ndarray, 
                 target_uncertainties: np.ndarray, 
                 Model: any,
                 n_samples: int = 1000, 
                 **kwargs
                 ) -> dict:
    """
    Monte Carlo estimation of uncertainties for regression coefficients.

    Parameters:
    - predictors (np.ndarray): predictor variables.
    - targets (np.ndarray): target values to train on / predict.
    - Model (class): Model class to use for fitting.
    - d (int): Number of groups to split the data into.
    - kwargs: Additional arguments for the solver.

    Returns:
    - coefs (np.ndarray): Coefficients for each fold.
    - coefs_mean (np.ndarray): Mean of coefficients.
    - coefs_std (np.ndarray): Standard deviation of coefficients.
    """

    shape_ = (n_samples, targets.shape[0])
    MC_targets = np.random.normal(targets, target_uncertainties, shape_)
    Model.set_X_train(predictors)

    coefs = []
    for i in tqdm(range(n_samples), desc="Monte Carlo Sampling"):
        Model.train(MC_targets[i])
        coefs.append(Model.coefficients)

    coefs = np.array(coefs)
    return {'sampled_coefs':coefs, 
            'coefs_mean': np.mean(coefs, axis=0), 
            'coefs_var': np.var(coefs, axis=0)}



def mcmc_sampling():
    """
    Markov Chain Monte Carlo estimation of uncertainties for regression coefficients.
    """
    pass


def get_delete_1_jk_var(
                        estimators: np.ndarray
                       ) -> np.ndarray:
    n = len(estimators)
    mean_ = np.mean(estimators, axis=0)
    var_ = (n-1)/n * np.sum((estimators - mean_)**2, axis=0)
    return var_


def get_delete_d_jk_var(
                        n_data: int,
                        estimators: np.ndarray, 
                        n_groups: int,
                        n_delete:int, 
                       )-> np.ndarray:
    mean_ = np.mean(estimators, axis=0)
    var_ = ( n_data - n_delete ) / ( n_delete * n_groups ) * np.sum((estimators - mean_)**2, axis=0)
    return var_


def get_grouped_jk_var(
                       estimators: np.ndarray, 
                       n_groups: int
                      ) -> np.ndarray:
    mean_ = np.mean(estimators, axis=0)
    var_ = (n_groups-1)/n_groups * np.sum((estimators - mean_)**2, axis=0)
    return var_


def get_confidence_intervals_normal(
                                    full_estimator: np.ndarray, 
                                    var: np.ndarray, 
                                    alpha: float = 0.05
                                    ) -> tuple:
    """Compute confidence interval using the normal approximation."""
    se = np.sqrt(var)
    z = stats.norm.ppf(1 - alpha/2)
    return (full_est - z*se, full_estimator + z*se)


def get_confidence_intervals_tscore(
                                     full_estimator: np.ndarray, 
                                     var: np.ndarray, 
                                     dof: int, 
                                     alpha: float = 0.05
                                     ) -> tuple:
    """Compute confidence interval using the t-distribution."""
    se = np.sqrt(var)
    t_val = stats.t.ppf(1 - alpha/2, df=dof)
    return (full_est - t_val*se, full_estimator + t_val*se)