import streamlit as st
from keras.models import load_model
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def scale_function(training_set):
    sc = MinMaxScaler()
    training_set_scaled = sc.fit_transform(training_set)

    sc_predict = MinMaxScaler()
    sc_predict.fit_transform(training_set[:, 0:1])
    return training_set_scaled,sc_predict


def inverse_transforme(sc_predict,predi_fut_or_trai):
    
    return sc_predict.inverse_transform(predi_fut_or_trai)


@st.cache(allow_output_mutation=True)
def data_structure_creation (training_set_scaled, dataset_train,n_future,n_past):
    # Creating a data structure with 90 timestamps and 1 output
    X_train = []
    y_train = []

    for i in range(n_past, len(training_set_scaled) - n_future +1):
        X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
        y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train


@st.cache(allow_output_mutation=True)
def predtion_LSTM_model(x):
    model1 = load_model('LSTM_model.h5')
    y_pred = model1.predict(x)
    return y_pred

def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


@st.cache(allow_output_mutation=True)
def plot_prep(dataset_train,datelist_train,y_pred_future, y_pred_train,cols,n_future,n_past):
    # Generate list of sequence of days for predictions
    datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

    '''
    Remeber, we have datelist_train from begining.
    '''

    # Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
    datelist_future_ = []
    for this_timestamp in datelist_future:
        datelist_future_.append(this_timestamp.date())

    PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['temp']).set_index(pd.Series(datelist_future))
    PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['temp']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))

    # Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
    PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

    dataset_train = pd.DataFrame(dataset_train, columns=cols)
    dataset_train.index = datelist_train
    dataset_train.index = pd.to_datetime(dataset_train.index)

    return datelist_future_ , PREDICTIONS_FUTURE, PREDICTION_TRAIN, dataset_train


@st.cache(allow_output_mutation=True) 
def func_plot(PREDICTIONS_FUTURE,PREDICTION_TRAIN,dataset_train):
    # Plot parameters
    fig, ax = plt.subplots(figsize=(14, 5))

    START_DATE_FOR_PLOTTING = '2021-10-17'

    ax.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['temp'], color='r', label='Predicted Temperature')

    ax.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['temp'], color='orange', label='Training Predictions')
    ax.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['temp'], color='b', label='Actual Temperature')

    ax.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

    ax.grid(which='major', color='#cccccc', alpha=0.5)

    ax.legend(shadow=True)
    ax.set_title('Predcitions and Acutal Temperature', family='Arial', fontsize=12)
    ax.set_xlabel('Timeline', family='Arial', fontsize=10)
    ax.set_ylabel('Temperature Value', family='Arial', fontsize=10)

    ax.tick_params(axis='x', labelrotation=45)
    #st.pyplot(fig)
    return fig
    
    