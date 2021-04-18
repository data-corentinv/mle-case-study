from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple, Union
from mlflow.pyfunc import PythonModel
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
logger = logging.getLogger(__name__)


class MultiModel(PythonModel, BaseEstimator, RegressorMixin):
    """
    Wrapper of multiple clones of a given estimator. Each clone differs only by:
        - the boostrap sample it is trained on
        - its random state (if any).
    Inherits BaseEstimator and RegressorMixin so as to fit into
    Inherits from PythonModel so as to be saved with MLflow.
    sklearn pipelines and sklearn clone method.

    Attributes
    ----------
    estimator : sklearn.BaseEstimator
        Any scikit-learn estimator.
    n : int
        Number of perturbed estimators.
    estimators : list
        List of fitted estimators.
    """

    def __init__(self, estimator: Optional[BaseEstimator] = None, n_models: int = 10) -> None:
        """
        Initialize the wrapper model.

        Parameters
        ----------
        estimator : BaseEstimator, optional
            Any sklearn model having a random_state attribute, by default None.
        n : int, optional
            Number of clones to maintain, by default 10.
        """
        self.n_models = n_models
        self.estimator = estimator
        logger.info(f'Instantiate {n_models} models of type:\n{estimator}')

    def _bootstrap(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        random_state: int = 1
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.array]]:
        """
        Generate a bootstrap from input arrays.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.
        y : Optional[pd.DataFrame] of shape (n_samples, 1)
            Training labels, by default None.
        random_state : int
            Random seed for random sampling.

        Returns
        -------
        tuple
            Bootstraps of X and y.
        """
        X_bootstrap = X.sample(frac=1, replace=True, random_state=random_state)
        if y is not None:
            y_bootstrap = y.sample(frac=1, replace=True, random_state=random_state)
            return X_bootstrap, np.ravel(y_bootstrap)
        else:
            return X_bootstrap

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> MultiModel:
        """
        Fit all clones and rearrange them into a list.
        The initial estimator is fit apart.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data.
        y : Optional[pd.Series] of shape (n_samples,)
            Training labels, by default None.

        Returns
        -------
        MultiModel
            The model itself.
        """
        self.single_estimator = clone(self.estimator)
        self.single_estimator.fit(X, np.ravel(y))
        self.estimators = []
        for random_state in range(self.n_models):
            e = clone(self.estimator)
            if hasattr(e, 'random_state'):
                e.set_params(random_state=random_state)
            X_bootstrap, y_bootstrap = self._bootstrap(X, y, random_state=random_state)
            e.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(e)
            logger.info(f'fit: X of shape {X.shape} on y - seed: {random_state}')
        return self

    def predict(self, context: Any, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for each clone and concatenate the results into a pandas dataframe.
        Predictions are bounded above zero.

        Parameters
        ----------
        context : Any
            Used by MLflow in some cases.
        X : pd.DataFrame of shape (n_samples, n_features)
            Prediction data.

        Returns
        -------
        pd.DataFrame
            Concatenation of each clone predictions.
        """
        preds = np.stack([self.estimators[i].predict(X) for i in range(self.n_models)], axis=1)
        preds = np.maximum(0, preds)
        preds = pd.DataFrame(
            preds,
            index=X.index,
            columns=['y_pred_{}'.format(i) for i in range(self.n_models)]
        )
        preds['y_pred_simple'] = self.single_estimator.predict(X)
        logger.info(f'predict: X of shape {X.shape}')
        return preds
