
#import os

#from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
#from pylab import rcParams
from matplotlib import pyplot as plt
#import numpy as np
import plotly.express as px

import pydeck as pdk 
from PIL import Image

#import time
#import ipywidgets
#import math

#from datetime import datetime
#from datetime import date
#from meteostat import Point, Hourly

from Map import display_map
from Weather_Data import data_generator,data_pre_processing
#from Prophet_model import load_Prophet_model, prediction_Prophet_model, plot_func
from LSTM_model import predtion_LSTM_model, plot_prep, data_structure_creation, inverse_transforme, scale_function, func_plot
#import pystan
#from prophet.plot import plot_plotly
#from plotly import graph_objs as go


#from siuda import _, filter
#from plotnine import *


st.set_page_config(page_title = 'Meteo Prediction',
                    page_icon = ":umbrella:",
                    layout ="wide")

# title of my app and write somthing in it 
st.title('My Meteo/Temperature Prediction')
#st.write('What do we expect? 🥶 or 🥵')

#image = Image.open('météo.jpg')
#st.image(image, caption='Meteo')



df = pd.read_json('Location History.json')

list_dico = df.values.tolist()
#list_dico

new_df = pd.DataFrame([ list_of_dico[0] for list_of_dico in list_dico])
#new_df

list_of_columns_to_keep = ['timestampMs',
                           'latitudeE7',
                           'longitudeE7',
                          'altitude']

clean_data = new_df[list_of_columns_to_keep].copy()
clean_data.columns = ['Date',
                      'Latitude',
                      'Longitude',
                     'Altitude']
#clean_data
#new_df

clean_data['Date'] = pd.to_datetime(clean_data['Date'], unit = 'ms')
clean_data['Latitude'] = clean_data['Latitude'].map(lambda X : X / 1E7)
clean_data['Longitude'] = clean_data['Longitude'].map(lambda X : X / 1E7)


## location extration
principal_loc = clean_data.groupby(['Longitude', 'Latitude','Altitude']).size().sort_values(ascending=False).reset_index()
#principal_loc
data = data_generator(principal_loc)

training_set, dataset_train,cols,datelist_train = data_pre_processing(data)

n_future = 60   # Number of days we want top predict into the future
n_past = 90     # Number of past days we want to use to predict the future


training_set_scaled,sc_predict = scale_function(training_set)

X_train, y_train = data_structure_creation (training_set_scaled, dataset_train,n_future,n_past)

predictions_future = predtion_LSTM_model(X_train[-n_future:])
predictions_train = predtion_LSTM_model(X_train[n_past:])

y_pred_future = inverse_transforme(sc_predict,predictions_future)
y_pred_train = inverse_transforme(sc_predict,predictions_train)

datelist_future_ , PREDICTIONS_FUTURE, PREDICTION_TRAIN, dataset_train = plot_prep(dataset_train,datelist_train,y_pred_future, y_pred_train,cols,n_future,n_past)

sided = st.sidebar

boutton0 = sided.button('Menu')
boutton1 = sided.button('raw data')
boutton2 = sided.button('raw data + little data process')
boutton3 = sided.button('Clean data')
boutton4 = sided.button('raw map')
boutton5 = sided.button('Locations where i have been the most')
boutton6 = sided.button('weather data')
boutton7 = sided.button('weather data after pre_processing features')
boutton8 = sided.button('LSTM')
#boutton9 = sided.button('Prophet')

if boutton0:
    # title of my app and write somthing in it 
    st.title('My Meteo/Temperature Prediction')
    st.write('What do we expect? 🥶 or 🥵')

    image = Image.open('météo.jpg')
    st.image(image, caption='Meteo')

elif boutton1:
    #clean_data.info()
    st.write('raw data')
    df.style.highlight_max(axis=0)
    st.dataframe(df.head(3))
elif boutton2:
    st.write('raw data + little data process')
    new_df.style.highlight_max(axis=0)
    st.dataframe(new_df.head(3))
elif boutton3:
    st.write('Clean data')
    clean_data.style.highlight_max(axis=0)
    st.dataframe(clean_data.head(3))
elif boutton4:
    st.pydeck_chart(display_map(clean_data))
elif boutton5:
    st.write('Location that i spend most of my time')
    principal_loc.style.highlight_max(axis=0)
    st.dataframe(principal_loc.head(3))
elif boutton6:
    st.write('weather data')
    data.style.highlight_max(axis=0)
    st.dataframe(data.head(3))
elif boutton7:
    training_set, dataset_train,cols,datelist_train = data_pre_processing(data)
    st.write('weather data after features pre_processing')
    dataset_train.style.highlight_max(axis=0)
    st.dataframe(dataset_train.head(3)) 
elif boutton8:
    fig = func_plot(PREDICTIONS_FUTURE,PREDICTION_TRAIN,dataset_train)
    st.pyplot(fig)
#elif boutton9:
#    st.markdown("---")
#    st.text("Prophet Visualisation")
#    st.image("download.png")
#    st.image("download (1).png")











#m = load_Prophet_model()
#forecast = prediction_Prophet_model(m)
#fig1, fig2 = plot_func(forecast,m)
#st.pyplot(fig1)
#st.pyplot(fig2)

#st.write('Location that i spend most of my time')
#principal_loc.style.highlight_max(axis=0)
#st.dataframe(principal_loc.head(3))
