"""FLAML sklearn wrapper — must match notebooks/04_modeling.ipynb for unpickling joblib models saved from the notebook."""

from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin

try:
    from flaml import AutoML as FLAMLAutoML

    FLAML_AVAILABLE = True
except Exception:  # pragma: no cover
    FLAMLAutoML = None
    FLAML_AVAILABLE = False


class FLAMLRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        time_budget=120,
        metric="rmse",
        estimator_list=None,
        seed=42,
        n_jobs=-1,
        task="regression",
        log_file_name="flaml.log",
        eval_method="cv",
        n_splits=3,
        split_ratio=None,
    ):
        self.time_budget = time_budget
        self.metric = metric
        self.estimator_list = estimator_list
        self.seed = seed
        self.n_jobs = n_jobs
        self.task = task
        self.log_file_name = log_file_name
        self.eval_method = eval_method
        self.n_splits = n_splits
        self.split_ratio = split_ratio
        self.automl_ = None

    def fit(self, X, y):
        if not FLAML_AVAILABLE:
            raise ImportError("FLAML is not installed.")
        self.automl_ = FLAMLAutoML()
        settings = {
            "time_budget": self.time_budget,
            "metric": self.metric,
            "task": self.task,
            "seed": self.seed,
            "n_jobs": self.n_jobs,
            "log_file_name": self.log_file_name,
            "eval_method": self.eval_method,
            "n_splits": self.n_splits,
        }
        if self.estimator_list is not None:
            settings["estimator_list"] = self.estimator_list
        if self.split_ratio is not None:
            settings["split_ratio"] = self.split_ratio
        self.automl_.fit(X_train=X, y_train=y, **settings)
        return self

    def predict(self, X):
        return self.automl_.predict(X)


def register_notebook_pickles() -> None:
    """Models saved from Jupyter reference __main__.FLAMLRegressorWrapper; map that for joblib.load."""
    import __main__

    setattr(__main__, "FLAMLRegressorWrapper", FLAMLRegressorWrapper)
