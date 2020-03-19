from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import sklearn as skl
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def predict_all(ticker): 
    get_data

## API call Stock data
def get_data(ticker):
    ts = TimeSeries(key='AJH8MF4OQ465LD6H',output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    data = data.sort_values(by = 'date')
    #Create a new dataframe for open price
    data = data.filter(['1. open'])
    #Convert the dataframe to a numpy array
    dataset = data.values

    return dataset

def prepare_data(dataset):
    #Get the number of rows to train the model on
    training_data_len = int( len(dataset) * .8 )
    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    #Create and scale the training data set
    train_data = scaled_data[0:training_data_len , :]
    #Split the data into x_train and y_train data sets
    X_train = []
    y_train = []
    for i in range(100, len(train_data)):
        X_train.append(train_data[i-100:i, 0])
        y_train.append(train_data[i, 0])
    #Convert the train data to numpy arrays 
    X_train, y_train = np.array(X_train), np.array(y_train)
    #Reshape to 3D from 2D to fit into LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train

def LSTM_model():
    #Define Model
    model = Sequential()
    #input later
    model.add(LSTM(100, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    #second layer
    model.add(LSTM(50, return_sequences= False))
    #third
    model.add(Dense(25))
    #Output
    model.add(Dense(1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)


def predict():
    #Test the model
    test_data = scaled_data[training_data_len - 100: , :]
    #Create the data sets x_test and y_test
    X_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(100, len(test_data)):
        X_test.append(test_data[i-100:i, 0])
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))

    #Get the models predicted price values 
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions

def plot(): 
    #Plot the data
    train = data[:training_data_len]
    tested = data[training_data_len:]
    tested['Predictions'] = predictions
    #Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Open Price USD ($)', fontsize=18)
    plt.plot(train['1. open'])
    plt.plot(tested[['1. open', 'Predictions']])
    plt.legend(['Train', 'Tested', 'Predictions'], loc='lower right')
    
    return plt.show()
