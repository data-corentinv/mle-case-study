""" Application - main script with mlflow implementation (etl part)
"""

import datetime
import os

import pandas as pd
from forecast.domain import transform, feature_engineering
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


def etl(raw_data_dir: str, preprocessed_data_dir: str = None, is_train: bool = True) -> pd.DataFrame:
    """ Extract transform and load data function (train or test set). 
    - Extract: turnover historical (train.csv, test.csv) and stores informations (bu_feat.csv)
    - Transform: merge and clean data
    - Load: save results in preprocessed data directory


    Parameters
    ----------
    raw_data_dir : str
        Path of data directory containing train.csv, test.csv, bu_feat.csv
    preprocessed_data_dir : str (default: None)
        Path of data for saving results of etl (default value is None, no data saved)
    is_train: bool (Default. True)
        Cleaning made on target only for the case of historical data processing

    Returns
    -------
    pd.DataFrame
        Dataframe cleaned
    """

    # extract
    if is_train:
        df_turnover = extract_turnover_history_csv(raw_data_dir)
    else: 
        df_turnover = extract_test_set_csv(raw_data_dir)

    df_stores = extract_stores_csv(raw_data_dir)

    # transform
    df = transform(df_turnover, 
                    df_stores, 
                    is_train=is_train)

    # load
    if preprocessed_data_dir:
        logger.info('Load transform data in preprocessed data directory')
        df.to_csv(f'{preprocessed_data_dir}/is_training_{str(is_train)}_preprocessed_data.csv')

    return df