import datetime
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

from forecast.domain.multimodel import MultiModel
from forecast.domain import transform, feature_engineering
from forecast.domain.forecast import simple_training
from forecast.infrastructure import extract_turnover_history_csv, extract_stores_csv, extract_test_set_csv
from forecast.settings import DATA_DIR_RAW

import mlflow
from mlflow.utils import mlflow_tags

import logging
logger = logging.getLogger(__name__)

import pdb

FEATURES = ['weekofyear_cos_1', 'weekofyear_sin_1', 'x', 'y', 'z', 'dpt_num_department_88', 
                'dpt_num_department_117', 'dpt_num_department_127', 'zod_idr_zone_dgr_3', 
                'zod_idr_zone_dgr_4', 'zod_idr_zone_dgr_6', 'zod_idr_zone_dgr_10', 'zod_idr_zone_dgr_35', 
                'zod_idr_zone_dgr_59', 'zod_idr_zone_dgr_72']

LIST_FEATURES_TO_DUMMY = ['dpt_num_department',"zod_idr_zone_dgr"]

def etl(raw_data_dir: str, preprocessed_data_dir: bool = None) -> pd.DataFrame:
    """ Extract transform and load data. Extract historical (train.csv, test.csv)
    and stores informations (bu_feat.csv)
    TODO: Adapt function for testset

    Parameters
    ----------
    raw_data_dir : str
        Path of data directory containing train.csv, test.csv, bu_feat.csv
    preprocessed_data_dir : str (default: None)
        Path of data for saving results of etl (default value is None, no data saved)

    Returns
    -------
    pd.DataFrame
        Dataframe cleaned
    """
    # Extract
    df_turnover = extract_turnover_history_csv(raw_data_dir)
    df_stores = extract_stores_csv(raw_data_dir)

    # Transform
    df = transform(df_turnover, 
                    df_stores)

    # Load <-> Save data
    if preprocessed_data_dir:
        logger.info('TODO: Data would be saved (not implemented)')

    return df

def validate_simple_model(df: pd.DataFrame):
    """ Train simple model
    """
    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='validate_simple_model') as run:
        logger.info(f"Start mlflow run - validate_simple_model - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'validate_simple_model')
        
        df_feat = feature_engineering(df, LIST_FEATURES_TO_DUMMY)
        
        mlflow.log_params(
            {
                'n_estimators': 20,
                'max_depath': 30,
                "dummy_features": LIST_FEATURES_TO_DUMMY,
                "nb_features": len(FEATURES)
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    
        model = RandomForestRegressor(n_estimators=20, max_depth=30, random_state=42)
        x_train, x_test, model = simple_training(df_feat, model, FEATURES)
        
        mlflow.log_metric(key="mae_test_set", value=round(x_test.MAE.mean(),1))
        mlflow.log_metric(key="mape_test_set", value=round(x_test.MAPE.mean(),1)*100)
        mlflow.log_metric(key="mae_train_set", value=round(x_train.MAE.mean(),1))
        mlflow.log_metric(key="mape_train_set", value=round(x_train.MAPE.mean(),1)*100)

        # log x_test
        x_test.to_csv('data/preprocessed/x_test.csv')
        mlflow.log_artifact('data/preprocessed/x_test.csv')

        # Plot results on test set
        fig, ax = plt.subplots(figsize=(20,10))
        ax.plot(range(x_test[:25].shape[0]), x_test[["turnover","y_pred_simple"]][:25])
        plt.grid(True)
        plt.savefig('./notebooks/images/tmp.png')
        mlflow.log_artifact("./notebooks/images/tmp.png")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='simple_model',
        )
       
        logger.info(f'mlflow.pyfunc.log_model:\n{model}')
            
    return None #TODO use get_run mlflow

def validate_multi_model(df: pd.DataFrame):
    """ Train simple model
    """
    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='validate_multi_model') as run:
        logger.info(f"Start mlflow run - validate_multi_model - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'validate_multi_model')
        
        df_feat = feature_engineering(df, LIST_FEATURES_TO_DUMMY)
        
        mlflow.log_params(
            {
                'n_estimators': 20,
                'max_depath': 30,
                "dummy_features": LIST_FEATURES_TO_DUMMY,
                "nb_features": len(FEATURES),
                'n_models':10
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    
        model = MultiModel(RandomForestRegressor(n_estimators=20, max_depth = 30), n_models=10)        
        x_train, x_test, model = simple_training(df_feat, model, FEATURES)
        
        mlflow.log_metric(key="mae_test_set", value=round(x_test.MAE.mean(),1))
        mlflow.log_metric(key="mape_test_set", value=round(x_test.MAPE.mean(),1)*100)
        mlflow.log_metric(key="mae_train_set", value=round(x_train.MAE.mean(),1))
        mlflow.log_metric(key="mape_train_set", value=round(x_train.MAPE.mean(),1)*100)

        # log x_test
        x_test.to_csv('data/preprocessed/x_test.csv')
        mlflow.log_artifact('data/preprocessed/x_test.csv')

        mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path='multi_model',
            code_path=[os.path.join('forecast', 'domain', 'multimodel.py')],
            conda_env={
                'channels': ['defaults', 'conda-forge'],
                'dependencies': [
                    'mlflow=1.8.0',
                    'numpy=1.17.4',
                    'python=3.7.6',
                    'scikit-learn=0.21.3',
                    'cloudpickle==1.3.0'
                ],
                'name': 'multi-model-env'
            }
        )
       
        logger.info(f'mlflow.pyfunc.log_model:\n{model}')
            
    return None #TODO use get_run mlflow

def main():
    # validate
    df = etl(DATA_DIR_RAW)
    validate_simple_model(df)
    #validate_multi_model(df)
    return 0

if __name__ == "__main__":
    main()