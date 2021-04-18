import pandas as pd
import logging
logger = logging.getLogger(__name__)

import pdb

def transform(
        df_turnover: pd.DataFrame, 
        df_stores: pd.DataFrame, 
        is_train: bool = True
        ) -> pd.DataFrame:
    """ Merge turnover data and store data. Clean features and 
    target with clean and clean_turnover functions


    Parameters
    ----------
    df_turnover : pd.DataFrame
        DataFrame containing daysid, dept, but (optional turnover)
    df_stores : pd.DataFrame
        DataFrame containing information of stores (latitude, long, postalcode, etc.)
    is_train : bool
        Boolean to launch target cleaning only on train set

    Returns
    -------
    pd.DataFrame
        Dataframe of merged and cleaned

    """

    df = merge_left(df_turnover, df_stores)
    logger.info(f'Merging turnover history with store informations done')

    df = clean(df)
    logger.info(f'Cleaning process done')

    if is_train: 
        df = clean_turnover(df)
        logger.info(f'Target cleaning (training process) done')

    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """ Created usefull new features (year and weekofyear feature)
    """
    df = new_feat_year(df)
    df = new_feat_weekofyear(df)
    return df.sort_values('day_id')

def clean_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean turnover (strange values such as negative turnover 
    and a specific outlier detected)
    """
    df = drop_negative_turnover(df)
    df = turnover_outliers_specific_imputation(df)
    return df

def drop_negative_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """ Drop negative turnover
    """
    return df.query(f'turnover >= 0')

def turnover_outliers_specific_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """ TODO: have to be change (create a more general outlier detection)
    """

    # compute mean turnover of departemnt, store in year 2013
    turnover_mean_value = df \
        .query(f"dpt_num_department == {88} and year == {2013} \
                    and weekofyear != {44} \
                    and but_num_business_unit == {30}") \
        .groupby(["weekofyear",'dpt_num_department'], as_index=False) \
        .agg({'turnover': sum}) \
        .turnover.mean()

    df.turnover.replace(1000000.0,turnover_mean_value, inplace=True)

    return df

def merge_left(df_turnover: pd.DataFrame, df_stores: pd.DataFrame) -> pd.DataFrame: 
    """ Merge left turnover and stores dataframe on but_num_business_unit

    Parameters
    ----------
    df_turnover : pd.DataFrame
        DataFrame containing daysid, dept, but (optional turnover)
    df_stores : pd.DataFrame
        DataFrame containing information of stores (latitude, long, postalcode, etc.)

    Returns
    -------
    pd.DataFrame
        Dataframe of stores
    """
    return pd.merge(df_turnover, df_stores, how='left', on='but_num_business_unit')


def new_feat_year(df: pd.DataFrame) -> pd.DataFrame:
    """ Create new feature YEAR from day_id

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing feature day_id 
    Returns
    -------
    df: pd.DataFrame
        Dataframe with new feature YEAR
    """
    return df.assign(year= df.day_id.dt.year)

def new_feat_weekofyear(df: pd.DataFrame) -> pd.DataFrame:
    """ Create new feature WEEKOFYEAR from day_id

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing feature day_id 
    Returns
    -------
    df: pd.DataFrame
        Dataframe with new feature WEEKOFYEAR
    """
    return df.assign(weekofyear= df.day_id.dt.isocalendar().week)
