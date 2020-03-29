from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, jsonify
from flask_pymongo import PyMongo
from pymongo import MongoClient


app = Flask(__name__)
#Use flask_pymongo to set up mongo connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/stocks_app"

mongo = PyMongo(app)
client = MongoClient(app.config['MONGO_URI'])
db = client.stocks

def predict():
        # Run all scraping functions and store results in dictionary
    week_prediction, predictions, tested, train = get_data()
    data = {
        "train": train,
        "tested": tested,
        "predictions": predictions,
        "week_predictions": week_prediction
        }
    db.stocks.insert_one(data)
    
    if __name__ == "__main__":
        # If running as script, print predicted data
        print(predict())



## API call Stock data
def get_data():
    ts = TimeSeries(key='AJH8MF4OQ465LD6H',output_format='pandas')
    data, meta_data = ts.get_daily(symbol='INX', outputsize='full')
    data = data.sort_values(by = 'date')
    #Create a new dataframe for open price
    data = data.filter(['1. open'])
    #Convert the dataframe to a numpy array
    dataset = data.values
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
    for i in range(50, len(train_data)):
        X_train.append(train_data[i-50:i, 0])
        y_train.append(train_data[i, 0])
    #Convert the train data to numpy arrays 
    X_train, y_train = np.array(X_train), np.array(y_train)
    #Reshape to 3D from 2D to fit into LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #Define Model
    model = Sequential()
    #input later
    model.add(LSTM(15, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    #second layer
    model.add(LSTM(10, return_sequences= False))
    #third
    model.add(Dense(5))
    #Output
    model.add(Dense(1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    #Test the model
    test_data = scaled_data[training_data_len - 50: , :]
    #Create the data sets x_test and y_test
    X_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(50, len(test_data)):
        X_test.append(test_data[i-50:i, 0])
    #Convert the data to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))
    #Get the models predicted price values 
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    # we added this -1, to grab just the "last day" 5031; because we only want one predicted 5032nd day's open price
    concatenated_prices = X_test[-1:, :, :]
    next_day_prediction = model.predict(concatenated_prices)
    how_many_days_in_the_future = 7
    prediction_for_days_in_future = []# next_day_prediction[0][0]]
    for i in range(how_many_days_in_the_future):
        concatenated_prices = np.concatenate([
            #------------------------------v we slice *off* the 50 days ago value; so we only have 49 days of data;
            concatenated_prices[-1:, 1:, :], 
            # Add in the new predicted value
            np.expand_dims(next_day_prediction, -1)
        ], axis=1)
        # Therefore we can predict last day
        next_day_prediction = model.predict(concatenated_prices)
        #Appending onto the list (1D) the [0][0]th value of a 2D array; so we have a list of predicted values in the future
        prediction_for_days_in_future.append(next_day_prediction[0][0])
    # inverse_transform them 
    week_prediction = scaler.inverse_transform(np.expand_dims(prediction_for_days_in_future, -1))
    train = data[:training_data_len]
    tested = data[training_data_len:]

    return week_prediction, predictions, tested, train


get_data()
