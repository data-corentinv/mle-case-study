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
    df = transform(df_turnover, df_stores)

    # Load <-> Save data
    if preprocessed_data_dir:
        logger.info('TODO: Data would be saved (not implemented)')

    return df

def train_multimodel(df: pd.DataFrame):
    """ Train simple model
    """
    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='train_multi_model') as run:
        logger.info(f"Start mlflow run - train_multi_model - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'train_multi_model')
        
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
        x_train, _, model = simple_training(df_feat, model, FEATURES, nb_week_prediction=1)
        
        mlflow.log_metric(key="mae_train_set", value=round(x_train.MAE.mean(),1))
        mlflow.log_metric(key="mape_train_set", value=round(x_train.MAPE.mean(),1)*100)

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
            
    return model #TODO use get_run mlflow

def train_simple_model(df: pd.DataFrame):
    """ Train simple model
    """
    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='train_simple_model') as run:
        logger.info(f"Start mlflow run - train_simple_model - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'train_simple_model')
        
        df_feat = feature_engineering(df, LIST_FEATURES_TO_DUMMY)
        
        mlflow.log_params(
            {
                'n_estimators': 20,
                'max_depath': 30,
                "dummy_features": LIST_FEATURES_TO_DUMMY,
                "nb_features": len(FEATURES),
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    
        model = RandomForestRegressor(n_estimators=20, max_depth = 30)  
        x_train, _, model = simple_training(df_feat, model, FEATURES, nb_week_prediction=1)
        
        mlflow.log_metric(key="mae_train_set", value=round(x_train.MAE.mean(),1))
        mlflow.log_metric(key="mape_train_set", value=round(x_train.MAPE.mean(),1)*100)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='simple_model',
        )
       
        logger.info(f'mlflow.pyfunc.log_model:\n{model}')
            
    return model #TODO use get_run mlflow

def predict(data_dir: str, 
            model: BaseEstimator):
    """ Make prediction based on model trained
    """
    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='prediction') as run:
        logger.info(f"Start mlflow run - prediction - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'prediction')
        
        df_test = extract_test_set_csv(data_dir)
        df_store = extract_stores_csv(data_dir)
        logger.info(f'shape of init test set {df_test.shape}')

        df = transform(df_test, df_store, is_train=False)
        df_feat = feature_engineering(df, list_features_to_dummy=LIST_FEATURES_TO_DUMMY)
        
        try:
            preds = model.predict(None, df_feat.set_index('day_id')[FEATURES])
        except (TypeError, ValueError): 
            preds = model.predict(df_feat.set_index('day_id')[FEATURES])
        
        try:
            preds.to_csv('data/preprocessed/preds.csv')
        except:
            pd.DataFrame({'predictions':preds}).to_csv('data/preprocessed/preds.csv')
        
        mlflow.log_artifact('data/preprocessed/preds.csv')
        mlflow.log_artifact(data_dir+'/test.csv')

    return preds

def prediction_based_on_log(df:pd.DataFrame, data_dir: str, logged_model: str):
    """ generate prediction on a new set
    ex. logged_model = './mlruns/0/bc5d7ba4f78145f9aac8e38ee4eade1a/artifacts/simple_model'
    """
    model = mlflow.pyfunc.load_model(logged_model)
    df_store = extract_stores_csv(data_dir)
    df_ = transform(df, df_store, is_train=False)
    df_ = feature_engineering(df_, list_features_to_dummy=LIST_FEATURES_TO_DUMMY)

    try:
        preds = model.predict(None, df_.set_index('day_id')[FEATURES])
        df = pd.concat([df.set_index('day_id'), preds],axis=1)
    except (TypeError, ValueError): 
        preds = model.predict(df_.set_index('day_id')[FEATURES])
        df.sort_values('day_id')['y_pred_simple'] = preds
    
    return df

def main():
    df = etl(DATA_DIR_RAW)
    model = train_simple_model(df)
    #model = train_multimodel(df)
    preds = predict(DATA_DIR_RAW, model=model)
    return 0

if __name__ == "__main__":
    main()
