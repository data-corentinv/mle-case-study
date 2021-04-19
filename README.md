# ML Engineer case study

## Data - inputs files

### Weekly store-department turnover 
train.csv.gz & test.csv.gz files contain weekly store-department turnover data (test.csv.gz does not contain the turnover feature).

### Stores information
bu_feat.csv.gz file contains some useful store information.

### Location
Data have been located in data/raw directory.

## Get strarted
### 1. Creating the virtual env 

The following command lines allow you to create the virtual env (.venv):

```
$ python3.8 -m venv /.venv
$ source activate.sh
```

### 2. Set up the virtual env

When running the program for the first time, you need to set up the virtual environment by running the following command lines:

Install the project's requirements:

```
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

For the next times you only need to activate the virtual environment:

```
$ source activate.sh
```

In order to use environment with jupyter, you can link it thanks this command :
```
$ python -m ipykernel install --user --name=decathlon-env
```

### 3. Jupyter notebooks

2 notebooks have been created : [01 - Preliminary questions & data exploration.md](notebooks/myexperiments/01\ -\ Preliminary\ questions\ &\ data\ exploration.md) and [02 - Forecast model.md](notebooks/myexperiments/02\ -\ Forecast\ model.md). There are located in notebooks/myexperiments directory.

To generated .ipynb files, you can use this command (jupytext): 

```
$ make notebooks
```

Then, you can run jupyter notebook :

```
$ jupyter notebook
```


## Outputs and model performance

### 1. Launch the model

```
$ make pipeline-validate
```

```
$ make pipeline-predict
```

### 2. Read the outputs

```
$ make run-webapp
```

### 3. Track model performance

```
$ make mlflow-ui
```


## Sphinx documentation

Project attached sphinx documentation. To generate it you can use this command :
```
$ make doc
```

Documentation of forecast project can be found here : notebooks/documentation

## Tests
### Data tests
> TODO
### Unit tests
> TODO

## Code structure

### DDD architecture (QM template)
<img src=notebooks/images/ddd.png />

### Details of structure

    ├── coverage
    ├── data
    │   ├── raw             <- Raw data
    │   ├── preprocessing   <- Intermediate data that has been cleaned and transformed.
    │   └── results         <- Data output of modelling (prediction for example)
    ├── doc                 <- Sphinx documentation of forecast project
    ├── forecast
    │   ├── application     <- Launch project in command line
    │   ├── domain          <- Core of project (Data preparation, feature engineering, ML algorithm)
    │   ├── infrastructure  <- Connect to data sources
    │   ├── interface       <- Expose results
    │   └── settings  
    ├── notebooks            
    │   ├── images          <- Schema for explanation (e.g. multimodel)
    │   └── myexperiments   <- Directory containing notebooks.
    ├── tests               <- tests (units tests, data tests)
    ├── .gitlab.yml         <- CI/CD gitlab
    ├── activate.sh         <- Bash script for activating virtual env
    ├── Makefile            <- Makefile with user friendly commands such as `make doc` or `make run-webapp`
    ├── README.md           <- The top-level README for developers using this project.
    └── requirements.txt    <- The requirements file for reproducing the python environment


## TODO:

- tests
- mlflow (get_run)
- gitlabci