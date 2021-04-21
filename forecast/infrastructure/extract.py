""" Infrastructure - connections to data (extract and load data)
"""

import pandas as pd

import logging
logger = logging.getLogger(__name__)

def extract_turnover_history_csv(data_dir: str) -> pd.DataFrame:
    """
    Load turnover history with information of stores for a given data source (train.csv)

    Parameters
    ----------
    data_dir : str
        Data directory path.

    Returns
    -------
    pd.DataFrame
        Table of turnover history
    """

    dtypes = {'but_num_business_unit': 'int', 
                'dpt_num_department': 'int', 
                'turnover': 'float'
                }

    parse_dates = ['day_id']

    df_turnover = pd.read_csv(data_dir+'/train.csv', 
                            dtype=dtypes, 
                            parse_dates=parse_dates)

    logger.info(f'Extract turnover history done')
    
    return df_turnover

def extract_test_set_csv(data_dir:str) -> pd.DataFrame: 
    """
    Load test set to make prediction (test.csv).

    Parameters
    ----------
    data_dir : str
        Data directory path.

    Returns
    -------
    pd.DataFrame
        Table of turnover history (store id, dept, turnover)
    """
  
    dtypes = {'but_num_business_unit': 'int', 
                'dpt_num_department': 'int'
                }

    parse_dates = ['day_id']

    df_turnover = pd.read_csv(data_dir+'/test.csv', 
                            dtype=dtypes, 
                            parse_dates=parse_dates)

    logger.info(f'Extract set for prediction done')
      
    return df_turnover

def extract_stores_csv(data_dir: str) -> pd.DataFrame:
    """
    Load store information for a given data source (bu_feat.csv).

    Parameters
    ----------
    data_dir : str
        Data directory path.

    Returns
    -------
    pd.DataFrame
        Dataframe containing information of stores (id, latitude, long, regions, zone)
    """
    dtypes = {  
            'but_num_business_unit': 'int', 
            'but_postcode': 'int',    
            "but_latitude":"float", 
            "but_longitude": "float", 
            "but_region_idr_region": "int", 
            "zod_idr_zone_dgr": "int"
            }

    df_stores =  pd.read_csv(data_dir+'/bu_feat.csv', 
                            dtype=dtypes)

    logger.info(f'Extract stores information done')

    return df_stores
