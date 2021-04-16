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
import plotly.graph_objects as go
```

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
```

## Load data 

```python
dtypes = {'but_num_business_unit': 'int', 'dpt_num_department': 'int', 'turnover': 'float', 
          'but_postcode': 'int',"but_latitude":"float", "but_longitude": "float", "but_region_idr_region": "int",
          "zod_idr_zone_dgr": "int"}
parse_dates = ['day_id']
```

```python
df_turnover = pd.read_csv('../data/raw/train.csv', dtype=dtypes, parse_dates=parse_dates)
df_stores =  pd.read_csv('../data/raw/bu_feat.csv', dtype=dtypes)
df_turnover.shape, df_stores.shape
```

```python
df = pd.merge(df_turnover, df_stores, how='left', on='but_num_business_unit')
df = df.assign(year= df.day_id.dt.year)
df = df.assign(weekofyear= df.day_id.dt.isocalendar().week)
del df_turnover, df_stores
df.head(2)
```

## Clean data

```python
df.but_region_idr_region.unique().shape, df.zod_idr_zone_dgr.unique().shape
```

```python
df.loc[df.turnover < 0, 'turnover'] = 0
```

```python
df \
    .query(f"dpt_num_department == {88}") \
    .groupby(["weekofyear",'year'], as_index=False) \
    .agg({'turnover': sum})\
    .pivot(index='weekofyear', columns='year', values='turnover') \
    .plot()
plt.grid(True)
plt.title(f'Dep {88}')
```

```python
turnover_mean_value = df \
    .query(f"dpt_num_department == {88} and year == {2013} and weekofyear != {44} and but_num_business_unit == {30}") \
    .groupby(["weekofyear",'dpt_num_department'], as_index=False) \
    .agg({'turnover': sum}) \
    .turnover.mean()

df \
    .turnover.replace(1000000.0,turnover_mean_value, inplace=True)

df \
    .query(f"dpt_num_department == {88}") \
    .groupby(["weekofyear",'year'], as_index=False) \
    .agg({'turnover': sum})\
    .pivot(index='weekofyear', columns='year', values='turnover') \
    .plot()

plt.grid(True)
plt.title(f'Dep {88}')
```

# Feature eng

```python
df['season'] = (df.day_id.dt.month%12 + 3) // 3
```

```python
for i in df.season.unique():
    print(f'{i} :', list(df \
        .query(f"season == {i}").sort_values("weekofyear") \
        .weekofyear.unique()))
    
#1 : hiver
#2 : printemps
#3: ete
#4: autonomne
```

```python
degree=1
omega = 2*np.pi*(df.weekofyear)/53
for i in range(1, degree + 1):
    df['weekofyear_cos_' + str(i)] = np.cos(i*omega)
    df['weekofyear_sin_' + str(i)] = np.sin(i*omega)
```

```python
df['x'] = np.cos(df['but_latitude']) * np.cos(df['but_longitude'])
df['y'] = np.cos(df['but_latitude']) * np.sin(df['but_longitude'])
df['z'] = np.sin(df['but_latitude']) 
```

```python
df.zod_idr_zone_dgr.unique().shape
```

# DataPrep

```python
features = ['day_id', "turnover", "dpt_num_department", "weekofyear_cos_1", "weekofyear_sin_1", "x", "y", "z","zod_idr_zone_dgr"]
df_train = df[features]
```

```python
df_train = pd.get_dummies(df_train, columns=['dpt_num_department',"zod_idr_zone_dgr"], drop_first=True)
df_train.head(2)
```

```python
x = df_train.drop(columns=['turnover'])
y = df[['day_id', 'turnover']]
x = x.set_index('day_id')
y = y.set_index('day_id')['turnover']
```

```python
model = RandomForestRegressor(n_estimators=10, random_state=42)
```

```python
date_test_set = list(df_train.day_id.drop_duplicates().sort_values(ascending=False)[:8].values)
```

```python
x_test = df_train[df_train.day_id.isin(date_test_set)].sort_values("day_id").drop(columns=['turnover']).set_index('day_id')
y_test = df_train[df_train.day_id.isin(date_test_set)].sort_values("day_id")['turnover']
df_test = df[df.day_id.isin(date_test_set)].sort_values("day_id")
```

```python
x_train = df_train[~df_train.day_id.isin(date_test_set)].sort_values("day_id").drop(columns=['turnover']).set_index('day_id')
y_train = df_train[~df_train.day_id.isin(date_test_set)].sort_values("day_id")['turnover']

```

```python
model.fit(x_train, y_train)
```

```python
preds_test = pd.DataFrame(
            model.predict(x_test),
            index=x_test.index,
            columns=['y_pred_simple']
        )
df_test = df_test.assign(preds_test = preds_test.y_pred_simple.values)
```

```python
# OVERVIEW PERF
mean_absolute_error(df_test.turnover, df_test.preds_test), mean_absolute_percentage_error(df_test.turnover, df_test.preds_test)
```

```python
# EXEMPLE POUR 1 MAGASIN

tmp = df_test \
    .query(f"but_num_business_unit == {64} and dpt_num_department_127 == {1}") \
    .filter(items=['day_id', 'turnover', 'preds_test']).sort_values('day_id')
tmp.plot('day_id', ['preds_test', 'turnover'])
plt.grid()
plt.title(f'{round(mean_absolute_error(tmp.turnover, tmp.preds_test),1), round(mean_absolute_percentage_error(tmp.turnover, tmp.preds_test),1)}')
tmp = df_test \
    .query(f"but_num_business_unit == {64} and dpt_num_department_88 == {1}") \
    .filter(items=['day_id', 'turnover', 'preds_test']).sort_values('day_id') 
tmp.plot('day_id', ['preds_test', 'turnover'])
plt.grid()
plt.title(f'{round(mean_absolute_error(tmp.turnover, tmp.preds_test),1), round(mean_absolute_percentage_error(tmp.turnover, tmp.preds_test),1)}')

tmp = df_test \
    .query(f"but_num_business_unit == {64} and dpt_num_department_117 == {1}") \
    .filter(items=['day_id', 'turnover', 'preds_test']).sort_values('day_id') 
tmp.plot('day_id', ['preds_test', 'turnover'])
plt.grid()
plt.title(f'{round(mean_absolute_error(tmp.turnover, tmp.preds_test),1), round(mean_absolute_percentage_error(tmp.turnover, tmp.preds_test),1)}')

tmp = df_test \
    .query(f"but_num_business_unit == {64} and dpt_num_department_127 == {0} and dpt_num_department_88 == {0} and dpt_num_department_117 == {0}") \
    .filter(items=['day_id', 'turnover', 'preds_test']).sort_values('day_id') 
tmp.plot('day_id', ['preds_test', 'turnover'])
plt.grid()
plt.title(f'{round(mean_absolute_error(tmp.turnover, tmp.preds_test),1), round(mean_absolute_percentage_error(tmp.turnover, tmp.preds_test),1)}')

```

## Apprentissage TimeSerieSplit

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
            columns=['y_pred_simple']
        )
    mae_fold = mean_absolute_error(y_fold_test, preds_fold_test)
    mape_fold = mean_absolute_percentage_error(y_fold_test, preds_fold_test)
    maes.append(mae_fold)
    mapes.append(mape_fold)
    preds = pd.concat([preds, preds_fold_test], sort=True)
    print(f'Fold {fold} -')
    print(f'train shape: [{x_fold_train.shape, x_fold_train.index.min(), x_fold_train.index.max()} - test shape: {x_fold_test.shape, x_fold_test.index.min(), x_fold_test.index.max()}]')
```

```python
#dummy weekofyear
maes, mapes
```

```python
# cos,sin weekofyear
maes, mapes
```

```python
## TEst multiModel
```

```python
import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple, Union
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin


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
model = MultiModel(RandomForestRegressor(n_estimators=10), n_models=10)
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
            columns=['y_pred_simple']
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
def compute_maes(y_true: pd.Series, y_pred: pd.DataFrame) -> List[float]:
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
compute_maes(y_fold_test, preds_fold_test)
```

```python
compute_mapes(y_fold_test, preds_fold_test)
```

```python
mapes
```

```python

def plotly_predictions(preds: pd.DataFrame, y: Optional[pd.Series] = None) -> go.Figure:
    """
    (Plotly) Plot predictions and true labels if any.

    Parameters
    ----------
    preds : pd.DataFrame
        Predictions.
    y : Optional[pd.Series]
        True labels, by default None.

    Returns
    -------
    go.Figure
        The figure to plot.
    """
    fig = go.Figure()
    columns = [col for col in preds.columns if col.startswith('y_pred')]
    mini = preds[columns].min(axis=1)
    maxi = preds[columns].max(axis=1)
    if 'y_pred_simple' in preds.columns:
        fig.add_trace(
            go.Scatter(
                x=preds.index,
                y=preds['y_pred_simple'],
                line_color='red',
                name='simple predictions'
            )
        )
    if len(columns) > 1:
        fig.add_trace(
            go.Scatter(
                x=mini.index,
                y=mini,
                fill=None,
                line_color='orange',
                line_width=0,
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=maxi.index,
                y=maxi,
                fill='tonexty',
                line_color='orange',
                line_width=0,
                name='multiple predictions'
            )
        )
    if y is not None:
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=y,
                line_color='dodgerblue',
                name='cash-in'
            )
        )
        print(f'plotly_predictions: target shape = {y.shape}')
    print(f'plotly_predictions: predictions shape = {preds.shape}')
    fig.update_layout(
        title='Food forecasting',
        xaxis_title='date',
        yaxis_title='dollars',
        font=dict(
            family="Computer Modern",
            size=18,
            color="#7f7f7f"
        )
    )
    return fig

```

```python
fig = plotly_predictions(preds, y=y_train)
```

```python
df_train
```

```python
preds
```

```python
x
```

```python

```
