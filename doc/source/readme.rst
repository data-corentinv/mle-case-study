=======================
ML Engineer case study
=======================

Data - inputs files
-------------------

Weekly store-department turnover 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
train.csv.gz & test.csv.gz files contain weekly store-department turnover data (test.csv.gz does not contain the turnover feature).

Stores information
^^^^^^^^^^^^^^^^^^
bu_feat.csv.gz file contains some useful store information.

Location
^^^^^^^^
Data have been located in data/raw directory.

Get strarted
------------
1. Creating the virtual env 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following command lines allow you to create the virtual env (.venv):

```
$ python3.8 -m venv /.venv
$ source activate.sh
```

2. Set up the virtual env
^^^^^^^^^^^^^^^^^^^^^^^^^

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

3. Jupyter notebooks
^^^^^^^^^^^^^^^^^^^^

2 notebooks have been created : 
    - 01 - Preliminary questions & data exploration.md
    - 02 - Forecast model.md](notebooks/myexperiments/02\ -\ Forecast\ model.md 
    
There are located in notebooks/myexperiments directory.

To generated .ipynb files, you can use this command (jupytext): 

```
$ make notebooks
```

Then, you can run jupyter notebook :

```
$ jupyter notebook
```

4. Expose results
^^^^^^^^^^^^^^^^^^^^
Launch streamlit webapp.

```
$ make run-webapp
```

Model performance
-----------------

1. Launch the model
^^^^^^^^^^^^^^^^^^^^
Launch validation and prediction.

```
$ make pipeline
```

2. Track model performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Launch mlflow application for tracking algorithm performance.

```
$ make mlflow-ui
```


Sphinx documentation
--------------------

Project attached sphinx documentation. To generate it you can use this command :
```
$ make doc
```

Documentation of forecast project can be found here : notebooks/documentation

Tests
-----

Data tests
^^^^^^^^^^
> TODO: not implemented

Unit tests
^^^^^^^^^^
> TODO: not implemented

TODO
-----

    - tests (units, data)
    - mlflow (get_run implementation)
    - CI/CD