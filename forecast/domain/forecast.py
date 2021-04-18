import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import logging
logger = logging.getLogger(__name__)


def simple_training(x:pd.DataFrame,
                    model: BaseEstimator, 
                    features: list,
                    x_test:pd.DataFrame=None,
                    target: str = "turnover" ,
                    nb_week_prediction: int = 8
                    ) :
    """ simple training 

    Parameters
    ----------
        x: pd.DataFrame
            DataFrame used for training
        model: BaseEstimator
            Sklearn model (ex. RandomForestRegressor)
        features: list
            list of features for modeling
        x_test:pd.DataFrame (Default None)
            Test set (automatically generated if None)
        target: str (Default turnover)
            Name of target feature
        nb_week_prediction: int (Default 8)
            Nb week for prediction in case of automatic x_test generation
    Returns
    -------
    pd.DataFrame
        DataFrame ready for modeling
    """
    x = x.set_index('day_id')

    def custom_split_train_test(
                        x: pd.DataFrame,
                        nb_week_prediction:int
                ):
        """ simple customer split (keep 8 last date of dataframe)
        """
        filter_date_test_set = list(x.index.unique().sort_values(ascending=False)[:nb_week_prediction].values)
        x = x.sort_index()
        x_train = x[~x.index.isin(filter_date_test_set)]
        x_test = x[x.index.isin(filter_date_test_set)]
        return x_train, x_test

    if x_test is None:
        x_train, x_test = custom_split_train_test(x, nb_week_prediction=nb_week_prediction)
    else: 
        x_train = x
        x_test = x_test.set_index('day_id')

    model.fit(x_train[features], x_train[target])

    try:
        x_test = pd.concat([x_test, model.predict(None, x_test[features])],axis=1)
        x_train = pd.concat([x_train, model.predict(None, x_train[features])],axis=1)
    except (TypeError, ValueError):
        x_test['y_pred_simple'] = model.predict(x_test[features])
        x_train['y_pred_simple'] = model.predict(x_train[features])

    
    logger.info('Test set computation:')
    x_test = compute_mae_mape_per_points(x_test)
    logger.info('Train set computation:')
    x_train = compute_mae_mape_per_points(x_train)

    return x_train, x_test, model

def cross_validate(df: pd.DataFrame, 
                    model: BaseEstimator,
                    features: list,
                    target: str='target', 
                    nb_week_prediction: int = 8,
                    n_fold:int=3):

    """ TimeSeriesSplit cross validation training


    Parameters
    ----------
        df: pd.DataFrame
            DataFrame used for training
        model: BaseEstimator
            Sklearn model (ex. RandomForestRegressor)
        features: list
            list of features for modeling
       target: str (Default turnover)
            Name of target feature
        nb_week_prediction: int (Default 8)
            Nb week for prediction in case of automatic x_test generation
    Returns
    -------
    pd.DataFrame
        DataFrame ready for modeling

    """
    cv = TimeSeriesSplit(n_fold, test_size=10059) # ~ size of test set

    for fold, (train_index, test_index) in enumerate(cv.split(df)):
        x_fold_train, x_fold_test = df.iloc[train_index], df.iloc[test_index]
        model_fold = clone(model)
        
        x_fold_train, x_fold_test, model = simple_training(x=x_fold_train, 
                        x_test = x_fold_test, 
                        features = features, 
                        model = model_fold, 
                        nb_week_prediction = nb_week_prediction)

        logger.info(f'Fold {fold} -')
        logger.info(f"""Train shape, start/end date: [{x_fold_train.shape, 
                                                        str(x_fold_train.index.min()), 
                                                        str(x_fold_train.index.max())} 
                        - test shape, start/end date: {x_fold_test.shape, 
                                                        str(x_fold_test.index.min()), 
                                                        str(x_fold_test.index.max())}]""")

    return x_fold_train, x_fold_test, model


def compute_mae_mape_per_points(x:pd.DataFrame, 
                            prediction_name:str="y_pred_simple"
                            ) -> pd.DataFrame:
    """ Compute 2 metrics : Mean Absolute Percentage Error (MAPE)
    and Mean Absolute Error (MAE)
    """

    # MAPE
    x['MAPE'] = abs(x['turnover'] - x[prediction_name]) / x['turnover']
    x['MAPE'].fillna(0, inplace=True)
    x['MAPE'] = x['MAPE'].replace(np.inf, 0) # TODO: exlusion draft

    # MAE
    x['MAE'] = abs(x['turnover'] - x[prediction_name])

    mean_mape = round(x.MAPE.mean(),1)
    mean_mae = round(x.MAE.mean(),1)

    logger.info(f'Metrics computed: mape {mean_mape}, mae {mean_mae}')

    return x