import numpy as np

class TimeSeriesObject:
    def __init__(self, times, observations, uncertainties):
        """
        Initialize the time series object.

        Parameters:
        - times: dict of np.ndarray
            Times for each filter (keyed by filter name).
        - observations: dict of np.ndarray
            Observations for each filter.
        - uncertainties: dict of np.ndarray
            Uncertainties for each filter.
        """
        self.times = times
        self.observations = observations
        self.uncertainties = uncertainties
        self.summary = self._compute_summary()
        self.N = self.summary['n_points']

    
    def _compute_summary(self):
        """Compute summary statistics for the time series."""
        summary = {
            "time_base": self.times.max() - self.times.min(),
            "n_points": len(self.times),
            "median_time_step": np.median(np.diff(np.sort(self.times)))
        }
        return summary

    def __repr__(self):
        """String representation for debugging."""
        summary_str = "\n".join([f"{stat}: {val}" for stat, val in self.summary.items()])
        return f"TimeSeriesObject with properties:\n{summary_str}"
