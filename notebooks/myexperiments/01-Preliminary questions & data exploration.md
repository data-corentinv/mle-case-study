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

# 01 - Preliminary questions & data exploration
    Author: corentinv

```python
import sys
sys.path.append('../../')

import pandas as pd
pd.set_option('display.min_rows', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('max_colwidth', 400)

import matplotlib.pyplot as plt
from folium import plugins
import folium

from forecast.application import etl
import forecast.settings as settings
from forecast.infrastructure.extract import extract_test_set_csv

import yaml
import logging
import logging.config
with open(settings.LOGGING_CONFIGURATION_FILE, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))
logging.getLogger('matplotlib.font_manager').disabled = True

```

# Loading and cleaning data (train.csv) 


```python
?etl
```

```python
df = etl(settings.DATA_DIR_RAW)
```

```python
df.head(2)
```

```python
# Missing values check
df.isna().sum()
```

## Questions


### a. Which department made the hightest turnover in 2016

```python
df \
.query("year == 2016") \
.groupby('dpt_num_department', as_index=False) \
.agg({'turnover': sum}) \
.sort_values('turnover', ascending=False)
```

### b. What are the top 5 week numbers (1 to 53) for department 88 in 2015 in terms of turnover over all stores ?

```python
df \
.query("year == 2015 and dpt_num_department == 88") \
.groupby('weekofyear', as_index=False) \
.agg({'turnover': sum}) \
.sort_values('turnover', ascending=False) \
.head(5)
```

## c. What was the top performer store in 2014

```python
df \
.query("year == 2014") \
.groupby('but_num_business_unit', as_index=False) \
.agg({'turnover': sum}) \
.sort_values('turnover', ascending=False) \
.head(10)
```

## d.e. Base on sales can you guess what kind of sport represents departement 73 & 117?

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

```python
# Top performer store for dept 73

df \
.query("dpt_num_department == 73") \
.groupby(['but_num_business_unit','but_postcode'], as_index=False) \
.agg({'turnover': sum}) \
.sort_values('turnover', ascending=False) \
.head(5)
```

```python
# Top performer stores for dept 117

df \
.query("dpt_num_department == 117") \
.groupby(['but_num_business_unit','but_postcode'], as_index=False) \
.agg({'turnover': sum}) \
.sort_values('turnover', ascending=False) \
.head(5)
```

    Ideas : 
        - spring sport for 73 : nautic sports (closed to sea)
        - winter sport for 117 : mountain sports (ski, snowboard) or winter clothes


## f. what other insights can you draw from the data ? Provide plots and figures if needed. (Opt)


### Fig1: Map of stores

```python
# Year selection 
year = 2014
# Drawing stores as points
is_draw_point = True
```

```python
# Create tempory dataframe
tmp = df \
.query(f"year == {year}") \
.groupby(['but_num_business_unit',"but_latitude","but_longitude"], as_index=False) \
.agg({'turnover': sum}) \
.sort_values("turnover", ascending=False)#\
```

```python
# Init map with a potential center stores in france
middle_lat = tmp['but_latitude'].median()
middle_lon = tmp['but_longitude'].median()
m = folium.Map(location=[middle_lat, middle_lon], zoom_start=6)

# mark each station as a point
if is_draw_point:
    for index, row in tmp.iterrows():
        folium.CircleMarker([row['but_latitude'], row['but_longitude']],
                            radius=5,
                            popup= row['but_num_business_unit'],
                            fill_color="#3db7e4", # divvy color
                           ).add_to(m)

# convert to (n, 2) nd-array format for heatmap
tmp["heat_map_weights_col"] = \
                    tmp["turnover"] / tmp["turnover"].sum()
stores = tmp[['but_latitude', 'but_longitude','turnover']].values

# plot heatmap
m.add_child(plugins.HeatMap(stores, radius=20))
m
```

```python
del tmp, stores, m
```

### Fig 2:  zone / regions of stores

```python
df \
.groupby('zod_idr_zone_dgr', as_index=False) \
.agg({'turnover': "mean"}) \
.plot('zod_idr_zone_dgr', kind='barh', title='Turnover for differents zones')
plt.grid(True)
```

```python
df \
.groupby('but_region_idr_region', as_index=False) \
.agg({'turnover': "mean"}) \
.plot('but_region_idr_region', kind='barh', title='Turnover for differents regions', figsize=(10,8))
plt.grid(True)
```

### Fig3: Plot turnover for differents departments and year

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
