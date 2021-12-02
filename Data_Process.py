import streamlit as st
import pandas as pd
from Weather_Data import data_generator,data_pre_processing
#from LSTM_Model import predtion_LSTM_model, plot_prep, data_structure_creation, inverse_transforme, scale_function, func_plot

@st.cache(allow_output_mutation=True)
def Data_cleaning ():
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

    return clean_data, df, new_df