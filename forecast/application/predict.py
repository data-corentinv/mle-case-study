""" Application - main script with mlflow implementation (predict part)
"""

import datetime
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

from forecast.domain.multimodel import MultiModel
from forecast.domain import transform, feature_engineering
from forecast.domain.forecast import simple_training, compute_mae_mape_per_points
from forecast.infrastructure import extract_turnover_history_csv, extract_stores_csv, extract_test_set_csv
from forecast.application import etl
import forecast.settings as settings
from forecast.settings import FEATURES, LIST_FEATURES_TO_DUMMY

import mlflow
from mlflow.utils import mlflow_tags
from forecast.application.mlflow_utils import mlflow_log_pandas

import yaml
import logging
import logging.config
with open(settings.LOGGING_CONFIGURATION_FILE, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))
logger = logging.getLogger(__name__)

import pdb


def train_simple_model(df: pd.DataFrame,
                        list_features_to_dummy: list,
                        features: list,     
                        n_estimators: int = 20,
                        max_depth: int = 30
                        ):
    """ Train simple model
    """
    logger.info('Simple model training in progress')

    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='train_simple_model') as run:
        logger.info(f"Start mlflow run - train_simple_model - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'train_simple_model')
        
        df_feat = feature_engineering(df, list_features_to_dummy)
        
        mlflow.log_params(
            {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                "dummy_features": list_features_to_dummy,
                "nb_features": len(features),
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth = max_depth)
        df_feat = df_feat.set_index('day_id')
        logger.info('learning in progress')
        model.fit(df_feat[features], df_feat['turnover'])
        logger.info('learning done')

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='simple_model',
        )

        logger.info(f'mlflow.pyfunc.log_model:\n{model}')
            
    return model #TODO use get_run mlflow

def train_multi_model(df: pd.DataFrame, 
                        list_features_to_dummy: list,
                        features: list,     
                        n_estimators: int = 20,
                        max_depth: int = 30, 
                        n_models: int=10
                        ):
    """ TODO Train mutli model
    """
    logger.info('Multi model training in progress')
    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='train_multi_model') as run:
        logger.info(f"Start mlflow run - train_multi_model - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'train_multi_model')
        
        df_feat = feature_engineering(df, list_features_to_dummy)
        
        mlflow.log_params(
            {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                "dummy_features": list_features_to_dummy,
                "nb_features": len(features),
                'n_models':n_models
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    
        model = MultiModel(RandomForestRegressor(n_estimators=n_estimators, max_depth = max_depth), n_models=n_models)     
        df_feat = df_feat.set_index('day_id')   
        model.fit(df_feat[features], df_feat['turnover'])

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
    logger.info(f'process validation multi model done')
            
    return model, n_models #TODO use get_run mlflow

def predict(data_dir: str, 
            list_features_to_dummy: list,
            features: list,  
            model: BaseEstimator, 
            n_models:int=None):
    """ Make prediction based on model trained
    """
    logger.info('Test set prediction in progress')
    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='prediction') as run:
        logger.info(f"Start mlflow run - prediction - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'prediction')
        
        df = etl(data_dir, is_train=False)
        df_test = extract_test_set_csv(data_dir)
        logger.info(f'shape of init test set {df_test.shape}')

        df_feat = feature_engineering(df, list_features_to_dummy=list_features_to_dummy)
        
        try:
            preds = model.predict(None, df_feat.set_index('day_id')[features])
            preds = pd.DataFrame(
                {
                    'y_pred_simple': preds['y_pred_simple'],
                    'y_pred_min': preds[[f'y_pred_{i}' for i in range(1,n_models)]].min(axis=1),
                    'y_pred_max': preds[[f'y_pred_{i}' for i in range(1,n_models)]].max(axis=1)
                }
            )
            logger.info('concatenage init test set with prediction')
            df_test = df_test.set_index('day_id')
            df_test = pd.concat([df_test.sort_index(), preds],axis=1)
        except (TypeError, ValueError): 
            logger.info('concatenage init test set with prediction')
            y_simple_prediction = model.predict(df_feat.sort_values('day_id')[features])
            df_test = df_test.sort_values('day_id') \
                            .assign(y_simple_prediction=y_simple_prediction)
         
        mlflow_log_pandas(df_test, 'test_set', 'test_set.csv')

    return 0

def main():
    df = etl(settings.DATA_DIR_RAW)
    model = train_simple_model(
                        df,
                        list_features_to_dummy = LIST_FEATURES_TO_DUMMY,
                        features = FEATURES,
                        n_estimators = 20,
                        max_depth = 30
                        )

    preds = predict(
                    settings.DATA_DIR_RAW, 
                    list_features_to_dummy = LIST_FEATURES_TO_DUMMY,
                    features = FEATURES,
                    model = model
                    )

    model, n_models = train_multi_model(
                        df,
                        list_features_to_dummy =  LIST_FEATURES_TO_DUMMY,
                        features = FEATURES,     
                        n_estimators = 20,
                        max_depth = 30, 
                        n_models = 3
                        )

    preds = predict(
                    settings.DATA_DIR_RAW, 
                    list_features_to_dummy = LIST_FEATURES_TO_DUMMY,
                    features = FEATURES,
                    model= model, 
                    n_models = n_models
                    )
    return 0

if __name__ == "__main__":
    main()
