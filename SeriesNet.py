# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:45:30 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, Input, Add, Activation, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import LeakyReLU, ELU
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error

  
# use mid to create a single price point and then make OHLC sampled data
def sampled_dframe(dframe, sample_code):
    mid_data=dframe['Mid']
    sampled_data=mid_data.resample(sample_code).ohlc()
    return sampled_data

def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def f(input_):
        residual =    input_
        
        layer_out =   Conv1D(filters=nb_filter, kernel_size=filter_length, 
                      dilation_rate=dilation, 
                      activation='linear', padding='causal', use_bias=False,
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(input_)
                    
        layer_out =   Activation('selu')(layer_out)
        
        skip_out =    Conv1D(1,1, activation='linear', use_bias=False, 
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)
        
        network_in =  Conv1D(1,1, activation='linear', use_bias=False, 
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)
                      
        network_out = Add()([residual, network_in])
        
        return network_out, skip_out
    
    return f

def DC_CNN_Model(length):
    inp = Input(shape=(length,1))
    
    l1a, l1b = DC_CNN_Block(32,2,1,0.001)(inp)    
    l2a, l2b = DC_CNN_Block(32,2,2,0.001)(l1a) 
    l3a, l3b = DC_CNN_Block(32,2,4,0.001)(l2a)
    l4a, l4b = DC_CNN_Block(32,2,8,0.001)(l3a)
    l5a, l5b = DC_CNN_Block(32,2,16,0.001)(l4a)
    l6a, l6b = DC_CNN_Block(32,2,32,0.001)(l5a)
    l6b = Dropout(0.6)(l6b) #dropout used to limit influence of earlier data
    l7a, l7b = DC_CNN_Block(32,2,64,0.001)(l6a)
    l7b = Dropout(0.6)(l7b) #dropout used to limit influence of earlier data

    l8 =   Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])
    
    l9 =   Activation('relu')(l8)

    l21 =  Conv1D(1,1, activation='linear', use_bias=False, 
           kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
           kernel_regularizer=l2(0.001))(l9)    

    model = Model(input=inp, output=l21)
    
    adam = optimizers.Adam(lr=0.00075, beta_1=0.9, beta_2=0.999, epsilon=None, 
                           decay=0.0, amsgrad=False)

    model.compile(loss='mae', optimizer=adam, metrics=['mse'])
    
    return model


def evaluate_timeseries(timeseries, predict_size):
    # timeseries input is 1-D numpy array
    # forecast_size is the forecast horizon
    
    timeseries = timeseries[~pd.isna(timeseries)]

    length = len(timeseries)-1

    timeseries = np.atleast_2d(np.asarray(timeseries))
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T 

    model = DC_CNN_Model(length)
    print('\n\nModel with input size {}, output size {}'.
                                format(model.input_shape, model.output_shape))
    
    model.summary()

    X = timeseries[:-1].reshape(1,length,1)
    y = timeseries[1:].reshape(1,length,1)
    
    model.fit(X, y, epochs=3000)
    
    pred_array = np.zeros(predict_size).reshape(1,predict_size,1)
    X_test_initial = timeseries[1:].reshape(1,length,1)
    #pred_array = model.predict(X_test_initial) if predictions of training samples required
    
    #forecast is created by predicting next future value based on previous predictions
    pred_array[:,0,:] = model.predict(X_test_initial)[:,-1:,:]
    for i in range(predict_size-1):
        pred_array[:,i+1:,:] = model.predict(np.append(X_test_initial[:,i+1:,:], 
                               pred_array[:,:i+1,:]).reshape(1,length,1))[:,-1:,:]
    
    return pred_array.flatten()


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

price=sampled['open']
predict_size=50
test_data=price[-predict_size:]
timeseries=price[:-predict_size]
len(test_data)

timeseries = timeseries[~pd.isna(timeseries)]
length = len(timeseries)-1
timeseries = np.atleast_2d(np.asarray(timeseries))
if timeseries.shape[0] == 1:
    timeseries = timeseries.T
    
inp = Input(shape=(length,1))
l1a, l1b = DC_CNN_Block(32,2,1,0.001)(inp)    
l2a, l2b = DC_CNN_Block(32,2,2,0.001)(l1a) 
l3a, l3b = DC_CNN_Block(32,2,4,0.001)(l2a)
l4a, l4b = DC_CNN_Block(32,2,8,0.001)(l3a)
l5a, l5b = DC_CNN_Block(32,2,16,0.001)(l4a)
l6a, l6b = DC_CNN_Block(32,2,32,0.001)(l5a)
l6b = Dropout(0.8)(l6b) 
l7a, l7b = DC_CNN_Block(32,2,64,0.001)(l6a)
l7b = Dropout(0.8)(l7b) 

l8 =   Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])

l9 =   Activation('relu')(l8)
l21 =  Conv1D(1,1, activation='linear', use_bias=False, 
       kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
       kernel_regularizer=l2(0.001))(l9)
model = Model(inputs=inp, outputs=l21)
adam = optimizers.Adam(lr=0.00075, beta_1=0.9, beta_2=0.999, epsilon=None, 
                       decay=0.0, amsgrad=False)
model.compile(loss='mse', optimizer=adam, metrics=['mse'])
print(model.input_shape, model.output_shape)
X = timeseries[:-1].reshape(1,length,1)
y = timeseries[1:].reshape(1,length,1)


model.fit(X, y, epochs=3000, verbose=0)

pred_array = np.zeros(predict_size).reshape(1,predict_size,1)
X_test_initial = timeseries[1:].reshape(1,length,1)
pred_array[:,0,:] = model.predict(X_test_initial)[:,-1:,:]
for i in range(predict_size-1):
    pred_array[:,i+1:,:] = model.predict(np.append(X_test_initial[:,i+1:,:], 
                           pred_array[:,:i+1,:]).reshape(1,length,1))[:,-1:,:]
result=pred_array.flatten()

print("mean squared error of prediction Field 1:", mean_squared_error(result,test_data) )