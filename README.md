# ML Engineer case study

## Data - inputs files

### Weekly store-department turnover 
train.csv.gz & test.csv.gz files contain weekly store-department turnover data (test.csv.gz does not contain the turnover).

### Store information
bu_feat.csv.gz file contains some useful store information.

### Data location
Raw data (_train.csv_, _test.csv_ and _bu_feat.csv_) have been located in _data/raw directory_.

Data preprocessing will be stored in _data/preprocessing_ and results in _data/results_ (cf. code structure section)

## Get strarted
### 1. Creating the virtual env 

Firstly you can create a virtual env.
The following command lines allow you to create it with python3.8 (.venv) and activate the environment :

```
$ python3.8 -m venv /.venv
$ source activate.sh
```

### 2. Set up the virtual env

Now you need to set up the virtual environment.
Install the project's requirements thanks to this command:

```
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

In order to use this environment in jupyter, you can link it thanks this command :
```
$ python -m ipykernel install --user --name=decathlon-env
```

### 3. Jupyter notebooks

2 notebooks have been created : 
- 01 - Preliminary questions & data exploration.md (preliminary question) and 
- 02 - Forecast model.md (forescast model explanation). 

There are located in the _notebooks/myexperiments_ directory.

To generate .ipynb files from .md, you can use this command (jupytext): 

```
$ make notebooks
```

Then, you just have to launch jupyter notebook :

```
$ jupyter notebook
```

### 4. Read the outputs

In order to visualize the results (predictions made), a streamlit application have been developed. 
This command line allow you to run the streamlit web application (based on the output of notebooks).

```
$ make run-webapp
```

<img src=notebooks/images/streamlit.png height=230 width=350/>

## Model pipeline and performance

### 1. Launch the model

A ML pipeline have been develop and configured with MLflow in order to track model performance and to save models and reuse it on other test set.

To launch the entire pipeline (validate and prediction) you can use this command:

```
$ make pipeline
```

If necessary, you can use "make pipeline-validate" command (to track model performance on train set and test set) or "make pipeline-prediction" command (to train model on the entire train set and to make predictions on test set).

### 2. Track model performance
This command launchs mlflow application for tracking algorithm performances: 

```
$ make mlflow-ui
```

<img src=notebooks/images/mlflow.png height=230 width=350 />

### 3. Get run MLflow - beta (work in progress)

If you want to make prediction on a new data set by using a model already train, copy the new test.csv in _data/new_ directory (have to be created).

Collect the _logged_model_ in mlflow web application and run this script :

```
$ python forecast/application/prediction_utils.py
```

## Sphinx documentation

Project has a  sphinx documentation. To generate it use this command :
```
$ make doc
```

Documentation of forecast project can be found here : notebooks/documentation.

<img src=notebooks/images/sphinx.png height=230 width=350 />

## Code structure

### DDD architecture (QM template)
<img src=notebooks/images/ddd.png/>

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

## Tests
### Data tests
> TODO: not implemented yet

### Unit tests
> TODO: not implemented yet 


## TODO:
    - intelligibility (pdb, ice, lime, shape)
    - tests (units, data)
    - mlflow (get_run implementation)
    - CI/CD