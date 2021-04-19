import pandas as pd

from forecast.domain import transform, feature_engineering
from forecast.infrastructure import extract_stores_csv, extract_test_set_csv
from forecast.application import etl
import forecast.settings as settings

import mlflow


def prediction_based_on_log(df: pd.DataFrame, 
                            data_dir: str, 
                            logged_model: str) -> pd.DataFrame:
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

if __name__ == "__main__":
    df = extract_test_set_csv(settings.data_dir + 'new/test.csv')
    res = prediction_based_on_log(df, 
                            settings.data_dir, 
                            './mlruns/0/bc5d7ba4f78145f9aac8e38ee4eade1a/artifacts/simple_model'
                            )
    res.to_csv(settings.data_dir + 'new/results_test.csv')