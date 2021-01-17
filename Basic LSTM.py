# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:58:04 2020

@author: RB
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

def sampled_dframe(dframe, sample_code):
    mid_data=dframe['Mid']
    sampled_data=mid_data.resample(sample_code).ohlc()
    return sampled_data

# IMPORT FROM PICKLE
# use mid to create a single price point and then make OHLC sampled data
combined_df = pd.read_pickle("combined_df.pkl")
print ("combined_df shape is ", combined_df.shape)
print(combined_df.head())

# prepare a sampled timeframe set
# Choose which time frame you want and proceed from here.
sampled=sampled_dframe(combined_df, '24H')
print ("sampled dataframe  dimensions", sampled.shape)
print("number of null values in sampled dataframe ", sampled.isna().sum())
sampled=sampled.dropna()
print ("updated sampled dataframe dimensions", sampled.shape)
print("number of null values in updated sampled dataframe ", sampled.isna().sum())
# create series for avg-price, high, low, close, percent_change, ln_change, ln_open_close
avg_price=(sampled['high']+sampled['low'])/2
open_price=sampled['open']
close_price=sampled['close']
high_price=sampled['high']
low_price=sampled['low']
ln_change=[]
ln_change.append(0)
for i in range(1,len(avg_price)):
    ln_change.append(np.log(avg_price[i]/avg_price[i-1]))
    
pct_change=[]
pct_change.append(0)
for i in range(1,len(avg_price)):
    pct_change.append((avg_price[i]-avg_price[i-1])/avg_price[i-1])
    
ln_open_close=[]
ln_open_close.append(0)
for i in range(1,len(avg_price)):
    ln_open_close.append(np.log(open_price[i]/close_price[i-1]))
    

print(" length of avg_price, high, low, open, close, pct_change, ln_change, ln_open_close series", len(avg_price), len(high_price), len(low_price), len(open_price), len(close_price), len(pct_change), len(ln_change), len(ln_open_close))

# create a dataframe of engineered features with same sampling frequency as sampled dataset
derived_price=pd.DataFrame(avg_price)
derived_price.columns=['avg_price']
cols=['ln_change', 'pct_change', 'ln_open_close']
derived=pd.concat([pd.Series(ln_change), pd.Series(pct_change), pd.Series(ln_open_close)], axis=1)
derived.columns=cols
derived.index=derived_price.index
derived=pd.merge(derived_price, derived, left_index=True, right_index=True)
derived.head()

# choose a segment of the whole data for training and testing. Use one of the above metrics for prediction
num_rows_train=2500
num_rows_test=600
start_row=1


# create the train data for all the direct and derived features
avg_price_train=avg_price[start_row-1:start_row+num_rows_train-1]
high_price_train=high_price[start_row-1:start_row+num_rows_train-1]
low_price_train=low_price[start_row-1:start_row+num_rows_train-1]
ln_open_close_train=ln_open_close[start_row-1:start_row+num_rows_train-1]
close_price_train=close_price[start_row-1:start_row+num_rows_train-1]
open_price_train=open_price[start_row-1:start_row+num_rows_train-1]
pct_change_train=pct_change[start_row-1:start_row+num_rows_train-1]
ln_change_train=ln_change[start_row-1:start_row+num_rows_train-1]

# create the test data for all the direct and derived features
avg_price_test=avg_price[start_row+num_rows_train:start_row+num_rows_train+num_rows_test]
high_price_test=high_price[start_row+num_rows_train:start_row+num_rows_train+num_rows_test]
low_price_test=low_price[start_row+num_rows_train:start_row+num_rows_train+num_rows_test]
ln_open_close_test=ln_open_close[start_row+num_rows_train:start_row+num_rows_train+num_rows_test]
close_price_test=close_price[start_row+num_rows_train:start_row+num_rows_train+num_rows_test]
open_price_test=open_price[start_row+num_rows_train:start_row+num_rows_train+num_rows_test]
pct_change_test=pct_change[start_row+num_rows_train:start_row+num_rows_train+num_rows_test]
ln_change_test=ln_change[start_row+num_rows_train:start_row+num_rows_train+num_rows_test]


# create dataset with features, labels and high-low of given data in the intervening prediction period. 
#returns data in array format and removes all timestamps
def create_dataset(data_series,look_back, predict_forward):       
    X_data=[]
    Y_data=[]
    Y_max=[]
    Y_min=[]
    for i in range(0,(len(data_series)-look_back-predict_forward+1)):
        X_data.append(data_series[i:(i+look_back)])
        Y_data.append(data_series[i+look_back+predict_forward-1])
    X_data=np.array(X_data)
    Y_data=np.array(Y_data)
    Y_max=np.array(Y_max)
    Y_min=np.array(Y_min)
    return (X_data, Y_data)

# look_back is the number of periods used for prediction
# predict_forward is the time in future that is predicted
look_back=20
predict_forward=1

# choose the feature to be trained
segment_series_train=low_price_train
segment_series_test=low_price_test
print (len(segment_series_train), len(segment_series_test))
print(segment_series_train)
print(segment_series_test)

# Create training and test data
(X_train,Y_train)=create_dataset(segment_series_train,look_back, predict_forward)
(X_test,Y_test,)=create_dataset(segment_series_test,look_back, predict_forward)
print("X_train, Y_train, X_test, Y_test: ", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

# Training of LSTM
model=Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.summary()
model.compile(optimizer = 'adam', loss = 'mse')
start_time=time.time()
model.fit(X_train, Y_train, epochs = 50, verbose=0)

X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
yhat = model.predict(X_test)
print("lookback :",look_back)
print("mean squared error of prediction :", mean_squared_error(yhat,Y_test) )






