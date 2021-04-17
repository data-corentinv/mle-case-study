---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.2
  kernelspec:
    display_name: decathlon-env
    language: python
    name: decathlon-env
---

```python
import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple, Union
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
```

```python
class MultiModel(BaseEstimator, RegressorMixin):
    """
    Wrapper of multiple clones of a given estimator. Each clone differs only by:
        - the boostrap sample it is trained on
        - its random state (if any).
    Inherits BaseEstimator and RegressorMixin so as to fit into
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
        print(f'Instantiate {n_models} models of type:\n{estimator}')

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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
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
            print(f'fit: X of shape {X.shape} on y - seed: {random_state}')
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
        print(f'predict: X of shape {X.shape}')
        return preds
```

```python
def compute_maes(y_true: pd.Series, y_pred: pd.DataFrame)b-> List[float]:
    """
    Return mean absolute errors of multi model given its predictions.

    Parameters
    ----------
    y_pred : pd.DataFrame
        Dataframe containing predicted labels (columns 'y_pred*').

    Returns
    -------
    list
        List of MAEs, one entry per model perturbation.
    """
    columns = [col for col in y_pred.columns if col.startswith('y_pred')]
    return [mean_absolute_error(y_true, y_pred[col]) for col in columns]
```

```python
def compute_mapes(y_true: pd.Series, y_pred: pd.DataFrame) -> List[float]:
    """
    Return mean absolute errors of multi model given its predictions.

    Parameters
    ----------
    y_pred : pd.DataFrame
        Dataframe containing predicted labels (columns 'y_pred*').

    Returns
    -------
    list
        List of MAEs, one entry per model perturbation.
    """
    columns = [col for col in y_pred.columns if col.startswith('y_pred')]
    return [mean_absolute_percentage_error(y_true, y_pred[col]) for col in columns]
```

```python
maes = []
mapes = []
preds = pd.DataFrame()
n_fold=3
cv = TimeSeriesSplit(n_fold, test_size=10136)
for fold, (train_index, test_index) in enumerate(cv.split(x_train, y_train)):
    x_fold_train, x_fold_test = x_train.iloc[train_index], x_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]
    model_fold = clone(model)
    model_fold.fit(x_fold_train, y_fold_train)
    try:
        preds_fold_test = model_fold.predict(None, x_fold_test)
    except (TypeError, ValueError):
        preds_fold_test = pd.DataFrame(
            model_fold.predict(x_fold_test),
            index=x_fold_test.index,
            columns=['ay_pred_simple']
        )
    mae_fold = compute_maes(y_fold_test, preds_fold_test)
    mape_fold = compute_mapes(y_fold_test, preds_fold_test)
    maes.append(mae_fold)
    mapes.append(mape_fold)
    preds = pd.concat([preds, preds_fold_test], sort=True)
    print(f'Fold {fold} -')
    print(f'train shape: [{x_fold_train.shape, x_fold_train.index.min(), x_fold_train.index.max()} - test shape: {x_fold_test.shape, x_fold_test.index.min(), x_fold_test.index.max()}]')
```

```python
compute_maes(y_fold_test, preds_fold_test)
```

```python
compute_mapes(y_fold_test, preds_fold_test)
```
