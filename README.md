# Temperature-Prediction-Meteo

### The main goal of this project is:

        •	To display my position (during this last 2 years) on a map (with pydeck) 
        •	Use an API to access an open weather and climate data (with Meteostat a Python library)
        •	use a machine learning algorithm (LSTM) to predict the weather


### Machine Learning model:

        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        #from keras.optimizers import Adam
        from tensorflow.keras.optimizers import Adam


        # Initializing the Neural Network based on LSTM
        model = Sequential()

        # Adding 1st LSTM layer
        model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))

        # Adding 2nd LSTM layer
        model.add(LSTM(units=32, activation='linear', return_sequences=False))

        # Adding Dropout
        model.add(Dropout(0.25))

        # Output layer
        model.add(Dense(units=1, activation='linear'))

        # Compiling the Neural Network
        model.compile(optimizer = Adam(learning_rate= 0.0001), loss='mean_squared_error')

        model.summary()
