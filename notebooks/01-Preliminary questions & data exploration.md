---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import matplotlib.pyplot as plt
```

```python
from folium import plugins
import folium
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

## Questions


### a. Which department made the hightest turnover in 2016

```python
df \
.query("year == 2016") \
.groupby('dpt_num_department', as_index=False) \
.agg({'turnover': sum})
```

```python
df \
.query("year == 2015 and dpt_num_department == 88") \
.groupby('weekofyear', as_index=False) \
.agg({'turnover': sum}) \
.sort_values('turnover', ascending=False) \
.head(5)
```

```python
df \
.query("year == 2014") \
.groupby('but_num_business_unit', as_index=False) \
.agg({'turnover': sum}) \
.sort_values('turnover', ascending=False) \
.head(10)
```

```python
fig, ax = plt.subplots(1,2, figsize=(15,5))

df \
.query("dpt_num_department in [73]") \
.groupby(['weekofyear'], as_index=False) \
.agg({'turnover': sum})\
.plot('weekofyear', 'turnover', ax=ax[0])
ax[0].grid(True)
ax[0].set_title('Dep 73')

df \
.query("dpt_num_department in [117]") \
.groupby(['weekofyear'], as_index=False) \
.agg({'turnover': sum})\
.plot('weekofyear', 'turnover', ax=ax[1])
ax[1].grid(True)
ax[1].set_title('Dep 117')
```

## Optional
Ideas :
- heatmap
- evolution (different department, year, stores)


```python
tmp = df \
.query("year == 2012") \
.groupby(['but_num_business_unit',"but_latitude","but_longitude"], as_index=False) \
.agg({'turnover': sum}) \
.sort_values("turnover", ascending=False)#\
#.head(10)
```

```python
# center store in france
middle_lat = tmp['but_latitude'].median()
middle_lon = tmp['but_longitude'].median()
m = folium.Map(location=[middle_lat, middle_lon], zoom_start=6)

# mark each station as a point
if False:
    for index, row in tmp.iterrows():
        folium.CircleMarker([row['but_latitude'], row['but_longitude']],
                            radius=15,
                            popup= row['but_num_business_unit'],
                            fill_color="#3db7e4", # divvy color
                           ).add_to(m)
```

```python
# convert to (n, 2) nd-array format for heatmap
tmp["heat_map_weights_col"] = \
                    tmp["turnover"] / tmp["turnover"].sum()
stores = tmp[['but_latitude', 'but_longitude','heat_map_weights_col']].values

# plot heatmap
m.add_child(plugins.HeatMap(stores, radius=20))
m
```

```python
for i in df.year.unique():
    tmp = df \
    .query(f"year == {i}") \
    .groupby(["weekofyear",'dpt_num_department'], as_index=False) \
    .agg({'turnover': sum})

    tmp \
    .pivot(index='weekofyear', columns='dpt_num_department', values='turnover') \
    .plot()
    plt.grid(True)
    plt.title(f'Year {i}')
```

```python
fig, ax = plt.subplots(1,1, figsize=(30,60))
df \
    .query("year == 2012") \
    .groupby(['but_num_business_unit'], as_index=False) \
    .agg({'turnover': sum})\
    .sort_values('turnover') \
    .plot("but_num_business_unit", kind='barh', ax=ax)
plt.grid(True)
```
