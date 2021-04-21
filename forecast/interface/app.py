""" Interface - User interface (streamlit)
"""

import numpy as np
import pandas as pd

import streamlit as st

from forecast.settings import DATA_DIR_OUTPUT
from forecast.interface.utils import read_df_train, read_df_test

def main():
    st.set_page_config(layout="wide")
    st.title('Expose results')

    df = read_df_train(DATA_DIR_OUTPUT)

    option = st.sidebar.selectbox(
        'Select a store: ',
        df['but_num_business_unit'].unique())

    option2 = st.sidebar.selectbox(
        'Select a department',
        df['dpt_num_department'].unique())

    st.write(f'You have selected : \n\n • Store n° {option} \n\n • Dept n° {option2}')

    st.header('I. Train set')

    start_date = st.sidebar.date_input('Start date', pd.to_datetime(df.day_id.min()))
    end_date = st.sidebar.date_input('End date', pd.to_datetime(df.day_id.max()))

    if start_date < end_date:
        st.write('• Start date: `%s`\n\n • End date:`%s`' % (start_date, end_date))
        tmp = df.query(f"""but_num_business_unit == {option}  \
                    and dpt_num_department == {option2}""")
        tmp_ = tmp
        tmp = tmp[pd.to_datetime(tmp.day_id) >= pd.to_datetime(start_date)]
        tmp = tmp[pd.to_datetime(tmp.day_id) <= pd.to_datetime(end_date)]

    else:
        st.error('Error: End date must fall after start date.')
        tmp = df.query(f"""but_num_business_unit == {option}  \
                    and dpt_num_department == {option2} \
                    """)
        tmp_ = tmp

    tmp.set_index('day_id', inplace=True)
    st.subheader('1. Table')
    st.table(tmp.head(8))
    st.write("\n")

    st.subheader('2. Chart line with predictions')
    st.line_chart(tmp[['turnover', 'y_pred_simple']], height= 500)

    st.subheader('3. Chart line with confidance interval')
    st.line_chart(tmp[['turnover', 'y_pred_min', 'y_pred_max']], height= 500)

    tmp['MEAN_MAE'] = tmp_['MAE'].mean()
    st.subheader('3. Chart line of MAE')
    st.line_chart(tmp[['MAE', 'MEAN_MAE']], height= 500)
    st.write('NB: MEAN_MAE is computed on all the history')

    tmp['MEAN_MAPE'] = tmp_['MAPE'].mean()
    st.subheader('4. Chart line of MAPE')
    st.line_chart(tmp[['MAPE', 'MEAN_MAPE']], height= 500)
    st.write('NB: MEAN_MAPE is computed on all the history')

    st.header('II. Test set')
    df_test = read_df_test(DATA_DIR_OUTPUT)

    tmp_test = df_test.query(f"""but_num_business_unit == {option}  \
                    and dpt_num_department == {option2}""")

    tmp_test.set_index('day_id', inplace=True)
    st.subheader('1. Table')
    st.table(tmp_test.head(8))
    st.write("\n")

    st.subheader('2. Chart line with predictions')
    st.line_chart(tmp_test[['y_pred_min', 'y_pred_max', 'y_pred_simple']], height= 500)

if __name__ == "__main__":
    main()
