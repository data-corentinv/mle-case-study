import pandas as pd

from forecast.domain import transform, feature_engineering
from forecast.infrastructure import extract_stores_csv, extract_test_set_csv
from forecast.application import etl
import forecast.settings as settings
from forecast.settings import FEATURES, LIST_FEATURES_TO_DUMMY

import mlflow

import yaml
import logging
import logging.config
with open(settings.LOGGING_CONFIGURATION_FILE, 'r') as f:
    logging.config.dictConfig(yaml.safe_load(f.read()))
logger = logging.getLogger(__name__)


import pdb

# TODO get params for a run
N_MODELS = 3

def prediction_based_on_log(df: pd.DataFrame, 
                            data_dir: str, 
                            logged_model: str
                            ) -> pd.DataFrame:
    """ generate prediction on a new set
    ex. logged_model = './mlruns/0/bc5d7ba4f78145f9aac8e38ee4eade1a/artifacts/simple_model'
    """

    model = mlflow.pyfunc.load_model(logged_model)
    df_store = extract_stores_csv(data_dir)
    df_ = transform(df, df_store, is_train=False)
    df_ = feature_engineering(df_, list_features_to_dummy=LIST_FEATURES_TO_DUMMY)

    try:
        preds = model.predict(df_.set_index('day_id')[FEATURES])
        preds = pd.DataFrame(
        {
                'y_pred_simple': preds['y_pred_simple'],
                'y_pred_min': preds[[f'y_pred_{i}' for i in range(1,N_MODELS)]].min(axis=1),
                'y_pred_max': preds[[f'y_pred_{i}' for i in range(1,N_MODELS)]].max(axis=1)
            }
        )
        logger.info('concatenage init test set with prediction')
        df = df.set_index('day_id')
        df = pd.concat([df.sort_index(), preds],axis=1)
    except (TypeError, ValueError): 
        logger.info('concatenage init test set with prediction')
        y_simple_prediction = model.predict(df_.sort_values('day_id')[FEATURES])
        df = df.sort_values('day_id') \
                        .assign(y_simple_prediction=y_simple_prediction)

    return df

if __name__ == "__main__":
    logger.info('You are trying to extract and run a trained model from mlflow')
    logger.info('Warning: implementation still in progress')
    logger.info('TODO use get_run to extract parameters')
    df = extract_test_set_csv(settings.DATA_DIR_NEW)
    res = prediction_based_on_log(df, 
                            settings.DATA_DIR_RAW, 
                            './mlruns/0/d3356832f6bb4df2978058d02d96df88/artifacts/multi_model'
                            )
    res.to_csv(settings.DATA_DIR_NEW + '/results_test.csv')
