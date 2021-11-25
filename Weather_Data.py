import streamlit as st
from datetime import datetime
#from datetime import date
from meteostat import Point, Hourly
import pandas as pd
import numpy as np

@st.cache(allow_output_mutation=True)
def data_generator(df):

    long_ = df['Longitude'].values[0]
    lat_ = df['Latitude'].values[0]
    alt_ = df['Altitude'].values[0]

    start = datetime(2000, 3, 19)
    end = datetime(2021, 9, 9)
    x=Point(lat_, long_, alt_)
    data = Hourly(x, start, end)
    data = data.fetch()
    #print()
    data = data.reset_index()
    return data

#@st.cache(allow_output_mutation=True)
def data_pre_processing(df):
    new_data = df.copy()
    new_data[['temp',
                'snow',
                'wdir',
                'wspd',
                'wpgt',
                'pres',
                'rhum',
                'dwpt',
                'prcp',
                'coco']] = new_data[['temp',
                'snow',
                'wdir',
                'wspd',
                'wpgt',
                'pres',
                'rhum',
                'dwpt',
                'prcp',
                'coco']].fillna(0)
    try_again = new_data.drop(columns=['tsun'])
    date_time = pd.to_datetime(try_again['time'],
                                 format='%d.%m.%Y %H:%M:%S')


    wpgt_kms = try_again.pop('wpgt')
    wspd_kms = try_again.pop('wspd')

    # Convert to radians.
    wd_rad = (try_again.pop('wdir')*np.pi) / 180

    # Calculate the wind x and y components.
    try_again['Wx'] = wspd_kms*np.cos(wd_rad)
    try_again['Wy'] = wspd_kms*np.sin(wd_rad)

    # Calculate the max wind x and y components.
    try_again['max Wx'] = wpgt_kms*np.cos(wd_rad)
    try_again['max Wy'] = wpgt_kms*np.sin(wd_rad)

    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24*60*60
    year = (365)*day

    try_again['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    try_again['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    try_again['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    try_again['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    dataset_train = try_again.drop(columns = ['time'])
    cols = list(dataset_train.columns)
    datelist_train = list(try_again['time'])

    dataset_train = dataset_train[cols].astype(str)

    dataset_train = dataset_train.astype(float)

    # Using multiple features (predictors)
    training_set = dataset_train.values

    return training_set, dataset_train,cols,datelist_train

    

