import streamlit as st
import pandas as pd

@st.cache
def read_df_train(data_dir:str):
    return pd.read_csv(data_dir+'/train_predictions.csv')

@st.cache
def read_df_test(data_dir:str):
    return pd.read_csv(data_dir+'/test_predictions.csv')
