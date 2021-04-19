import datetime
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

from forecast.domain.multimodel import MultiModel
from forecast.domain import transform, feature_engineering
from forecast.domain.forecast import simple_training, cross_validate
from forecast.infrastructure import extract_turnover_history_csv, extract_stores_csv, extract_test_set_csv
import forecast.settings as settings

import mlflow
from mlflow.utils import mlflow_tags
from forecast.application.mlflow_utils import mlflow_log_pandas, mlflow_log_pyplot

import yaml
import logging
import logging.config
with open(settings.LOGGING_CONFIGURATION_FILE, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))
logging.getLogger('matplotlib.font_manager').disabled = True
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

    # extract
    df_turnover = extract_turnover_history_csv(raw_data_dir)
    df_stores = extract_stores_csv(raw_data_dir)

    # transform
    df = transform(df_turnover, 
                    df_stores)

    # load
    if preprocessed_data_dir:
        logger.info('TODO: Data would be saved (not implemented)')

    return df

def validate_simple_model(df: pd.DataFrame, 
                        list_features_to_dummy: list,
                        features: list,     
                        n_estimators: int = 20,
                        max_depth: int = 30,                    
                        method: str = "simple",
                        n_fold: int=3):
    """ Cross validation of simple model

    df: pd.DataFrame
    method: str
        'simple' validation or 'timeseriesplit' (3 fold) validation
    """

    if not method in ['simple','timeseriesplit'] :
        logger.error('method validation unknown')
        return 1

    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='validate_simple_model') as run:
        logger.info(f"Start mlflow run - validate_simple_model - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'validate_simple_model')
        
        df_feat = feature_engineering(df, list_features_to_dummy)
        
        mlflow.log_params(
            {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                "dummy_features": list_features_to_dummy,
                "nb_features": len(features), 
                "validation_method": method
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        if method == 'simple':
            logger.info('simple validation method choosen')
            x_train, x_test, model = simple_training(df_feat, model, features)
        else: 
            logger.info('TimeSerieSplit validation method choosen (default 3 folds)')
            x_train, x_test, model = cross_validate(df_feat, model, features, n_fold=n_fold)
        
        mlflow.log_metric(key="mae_mean_test_set", value=round(x_test.MAE.mean(),1))
        mlflow.log_metric(key="mape_mean_test_set", value=round(x_test.MAPE.mean(),1)*100)
        mlflow.log_metric(key="mae_mean_train_set", value=round(x_train.MAE.mean(),1))
        mlflow.log_metric(key="mape_mean_train_set", value=round(x_train.MAPE.mean(),1)*100)

        # log x_test
        x_test.to_csv('data/preprocessed/x_test.csv')
        mlflow.log_artifact('data/preprocessed/x_test.csv')

        # Plot results on test set
        tmp_plot = x_test[:100]
        fig, ax = plt.subplots(figsize=(50,10))
        ax.plot(range(tmp_plot.shape[0]), tmp_plot[["turnover","y_pred_simple"]])
        plt.grid(True)
        plt.legend(['turnover', 'y_pred_simple'])
        mlflow_log_pyplot(fig, 'validation', 'plot_test_set.png')

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='simple_model',
        )
       
        logger.info(f'mlflow.pyfunc.log_model:\n{model}')

    logger.info(f'process validation simple model done')
    return 0 #TODO use get_run mlflow

def validate_multi_model(df: pd.DataFrame, 
                        list_features_to_dummy: list,
                        features: list,     
                        n_estimators: int = 20,
                        max_depth: int = 30, 
                        n_models: int=10,
                        method: str = "simple", 
                        n_fold: int = 3):
    """ Train simple model
    """

    if not method in ['simple','timeseriesplit'] :
        logger.error('method validation unknown')
        return 1
    
    mlflow_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name='validate_multi_model') as run:
        logger.info(f"Start mlflow run - validate_multi_model - id = {run.info.run_id}")
        mlflow.set_tag('entry_point', 'validate_multi_model')
        
        df_feat = feature_engineering(df, list_features_to_dummy)
        
        mlflow.log_params(
            {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                "dummy_features": list_features_to_dummy,
                "nb_features": len(features),
                'n_models':n_models, 
                "method_validation": method
            }
        )
        git_commit = run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    
        model = MultiModel(RandomForestRegressor(n_estimators=n_estimators, max_depth = max_depth), n_models=n_models)        
        
        if method == 'simple':
            logger.info('simple validation method choosen')
            x_train, x_test, model = simple_training(df_feat, model, features)
        else: 
            logger.info('TimeSerieSplit validation method choosen (default. 3 folds)')
            x_train, x_test, model = cross_validate(df_feat, model, features, n_fold=n_fold)
        
        x_train, x_test, model = simple_training(df_feat, model, features)
        
        mlflow.log_metric(key="mae_test_set", value=round(x_test.MAE.mean(),1))
        mlflow.log_metric(key="mape_test_set", value=round(x_test.MAPE.mean(),1)*100)
        mlflow.log_metric(key="mae_train_set", value=round(x_train.MAE.mean(),1))
        mlflow.log_metric(key="mape_train_set", value=round(x_train.MAPE.mean(),1)*100)

        # plot example results
        preds = pd.DataFrame(
                    { 
                        'y_pred_simple': x_test['y_pred_simple'],
                        'y_pred_min': x_test[[f'y_pred_{i}' for i in range(1,n_models)]].min(axis=1),
                        'y_pred_max': x_test[[f'y_pred_{i}' for i in range(1,n_models)]].max(axis=1),
                        'turnover': x_test['turnover']
                    }
                )

        tmp_plot = preds[:100]
        fig, ax = plt.subplots(figsize=(50,10))
        ax.plot(range(tmp_plot.shape[0]), tmp_plot["turnover"], label ='turnover' )
        ax.plot(range(tmp_plot.shape[0]), tmp_plot["y_pred_simple"], label='simple_pred')
        ax.fill_between(range(tmp_plot.shape[0]), tmp_plot["y_pred_min"], tmp_plot["y_pred_max"], alpha=0.2)
        plt.grid(True)
        plt.legend()
        mlflow_log_pyplot(fig, 'validation', 'plot_test_set.png')

        # log x_test
        mlflow_log_pandas(x_test, "test_validation", "x_test.csv")

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
            
    return 0 #TODO use get_run mlflow

def main():
    df = etl(settings.DATA_DIR_RAW)
    
    validate_simple_model(
                            df,
                            list_features_to_dummy = LIST_FEATURES_TO_DUMMY,
                            features = FEATURES,     
                            n_estimators= 20,
                            max_depth= 30,
                            method='simple'
                        )

    validate_simple_model(
                            df,
                            list_features_to_dummy = LIST_FEATURES_TO_DUMMY,
                            features = FEATURES,     
                            n_estimators= 20,
                            max_depth= 30,
                            method='timeseriesplit'
                        )
    
    validate_multi_model(
                            df,
                            list_features_to_dummy = LIST_FEATURES_TO_DUMMY,
                            features = FEATURES,     
                            n_estimators = 20,
                            max_depth = 30,
                            n_models = 10,
                            method='timeseriesplit'
                        )
    return 0

if __name__ == "__main__":
    main()