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
