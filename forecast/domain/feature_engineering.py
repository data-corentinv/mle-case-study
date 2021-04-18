import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

def feature_engineering(df: pd.DataFrame, list_features_to_dummy:list) -> pd.DataFrame:
    """ features enginnering (4 steps) :
        - 1/4 - Season feature 
        - Numerical feature of weekofyear (sin,cos)
        - x, y, z features from lat, long created
        - Dummies features of {list_features_to_dummy}

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing feature day_id 
    list_features_to_dummy: list
        List of features to dummy

    Returns
    -------
    df: pd.DataFrame
        Dataframe with new feature (season, sin/cos weekofyear, xyz, dummies)
    """
    logger.info(f'Feature engineering starts ...')
    df = new_feat_season(df) 
    logger.info(f'1/4 - Season feature have been created')
    df = new_feat_sin_cos_weekofyear(df)
    logger.info(f'2/4 - Numerical feature of weekofyear (sin,cos) created')
    df = new_feat_x_y_z_stores(df)
    logger.info(f'3/4 - x, y, z features from lat, long created')
    df = get_dummies_given_a_list(df, list_features_to_dummy=list_features_to_dummy)
    logger.info(f'4/4 - Dummies features of {list_features_to_dummy} created')
    logger.info(f'Feature engineering ended')

    return df

def get_dummies_given_a_list(df: pd.DataFrame, 
                            list_features_to_dummy: list
                            ) -> pd.DataFrame:
    """ Generate dummies from list of features

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing list of features
    list_features: list
        List of features to dummy
    Returns
    -------
    df: pd.DataFrame
        Dataframe with feature dummies
    """
    df = pd.get_dummies(df, columns=list_features_to_dummy, drop_first=True)
    return df

def new_feat_season(df: pd.DataFrame) -> pd.DataFrame:
    """ #TODO create estimated season featured (1: hiver, 2 : printemps, 3: ete, 4: autonomne)
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing feature day_id 
    Returns
    -------
    df: pd.DataFrame
        Dataframe with new feature SEASON
    """
    df['season'] = (df.day_id.dt.month%12 + 3) // 3
    return df

def new_feat_sin_cos_weekofyear(df: pd.DataFrame) -> pd.DataFrame:
    """ Generate numerical of from weekofyear feature with sin cos
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing feature day_id 
    Returns
    -------
    df: pd.DataFrame
        Dataframe with 2 additionnal features
    """
    degree=1
    omega = 2*np.pi*(df.weekofyear)/53
    for i in range(1, degree + 1):
        df['weekofyear_cos_' + str(i)] = np.cos(i*omega)
        df['weekofyear_sin_' + str(i)] = np.sin(i*omega)
    return df

def new_feat_x_y_z_stores(df:pd.DataFrame) -> pd.DataFrame:
    """ Generate x, y, z feature from latitude, longitude of stores

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing geographical stores features (lat, long)
    Returns
    -------
    df: pd.DataFrame
        Dataframe with 3 additionnal features x,y,z
    """
    df['x'] = np.cos(df['but_latitude']) * np.cos(df['but_longitude'])
    df['y'] = np.cos(df['but_latitude']) * np.sin(df['but_longitude'])
    df['z'] = np.sin(df['but_latitude']) 
    return df