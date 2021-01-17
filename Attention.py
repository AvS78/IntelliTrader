# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:09:10 2020

@author: RB
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

    
def sampled_dframe(dframe, sample_code):
    mid_data=dframe['Mid']
    sampled_data=mid_data.resample(sample_code).ohlc()
    return sampled_data

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

def sinusoidal_positional_encoding(
    inputs, num_units, zero_pad = False, scale = False
):
    T = inputs.get_shape().as_list()[1]
    position_idx = tf.tile(
        tf.expand_dims(tf.range(T), 0), [tf.shape(inputs)[0], 1]
    )
    position_enc = np.array(
        [
            [
                pos / np.power(10000, 2.0 * i / num_units)
                for i in range(num_units)
            ]
            for pos in range(T)
        ]
    )
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
    lookup_table = tf.convert_to_tensor(position_enc, tf.float32)
    if zero_pad:
        lookup_table = tf.concat(
            [tf.zeros([1, num_units]), lookup_table[1:, :]], axis = 0
        )
    outputs = tf.nn.embedding_lookup(lookup_table, position_idx)
    if scale:
        outputs = outputs * num_units ** 0.5
    return outputs


class Model:
    def __init__(
        self, seq_len, learning_rate, dimension_input, dimension_output
    ):
        self.X = tf.placeholder(tf.float32, [None, seq_len, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        x = self.X
        x += sinusoidal_positional_encoding(x, dimension_input)
        masks = tf.sign(self.X[:, :, 0])
        align = tf.squeeze(tf.layers.dense(x, 1, tf.tanh), -1)
        paddings = tf.fill(tf.shape(align), float('-inf'))
        align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.expand_dims(tf.nn.softmax(align), -1)
        x = tf.squeeze(tf.matmul(tf.transpose(x, [0, 2, 1]), align), -1)
        self.logits = tf.layers.dense(x, dimension_output)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate
        ).minimize(self.cost)
        
# IMPORT FROM PICKLE
# use mid to create a single price point and then make OHLC sampled data
combined_df = pd.read_pickle("combined_df.pkl")
print ("combined_df shape is ", combined_df.shape)
print(combined_df.head())
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


sampled['index'] = sampled.index
time_series = pd.to_datetime(sampled.iloc[:, 4]).tolist()        

# we are predicting for 50 forward periods
test=sampled.iloc[3069:,0:4]
sampled=sampled.iloc[:3069,0:4]

minmax = MinMaxScaler().fit(sampled.iloc[:, 0:4].astype('float32'))
sampled_log = minmax.transform(sampled.iloc[:, 0:4].astype('float32'))
sampled_log = pd.DataFrame(sampled_log)
print(sampled_log.head())

timestamp = 5
epoch = 100
future_day = 50

tf.reset_default_graph()
modelnn = Model(timestamp, 0.01, sampled_log.shape[1], sampled_log.shape[1])
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    total_loss = 0
    for k in range(0, (sampled_log.shape[0] // timestamp) * timestamp, timestamp):
        batch_x = np.expand_dims(
            sampled_log.iloc[k : k + timestamp].values, axis = 0
        )
        batch_y = sampled_log.iloc[k + 1 : k + timestamp + 1].values
        _, loss = sess.run(
            [modelnn.optimizer, modelnn.cost],
            feed_dict = {modelnn.X: batch_x, modelnn.Y: batch_y},
        )
        loss = np.mean(loss)
        total_loss += loss
    total_loss /= sampled_log.shape[0] // timestamp
    if (i + 1) % 10 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)
        
output_predict = np.zeros((sampled_log.shape[0] + future_day, sampled_log.shape[1]))
output_predict[0] = sampled_log.iloc[0]
upper_b = (sampled_log.shape[0] // timestamp) * timestamp

for k in range(0, (sampled_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits = sess.run(
        modelnn.logits,
        feed_dict = {
            modelnn.X: np.expand_dims(sampled_log.iloc[k : k + timestamp], axis = 0)
        },
    )
    output_predict[k + 1 : k + timestamp + 1] = out_logits

sampled_log.loc[sampled_log.shape[0]] = out_logits[-1]
time_series.append(time_series[-1] + timedelta(days = 1))

for i in range(future_day - 1):
    out_logits = sess.run(
        modelnn.logits,
        feed_dict = {
            modelnn.X: np.expand_dims(sampled_log.iloc[-timestamp:], axis = 0)
        },
    )
    output_predict[sampled_log.shape[0]] = out_logits[-1]
    sampled_log.loc[sampled_log.shape[0]] = out_logits[-1]
    time_series.append(time_series[-1] + timedelta(days = 1))
    
sampled_log = minmax.inverse_transform(sampled_log.values)
time_series = pd.Series(time_series).dt.strftime(date_format = '%Y-%m-%d').tolist()

f1_predict=sampled_log[3069:,0]
f2_predict=sampled_log[3069:,1]
f3_predict=sampled_log[3069:,2]
f4_predict=sampled_log[3069:,3]
f1_true=test.iloc[:,0].tolist()
f2_true=test.iloc[:,1].tolist()
f3_true=test.iloc[:,2].tolist()
f4_true=test.iloc[:,3].tolist()
print("time-lag: ", timestamp)
print("mean squared error of prediction Field 1:", mean_squared_error(f1_true,f1_predict) )
print("mean squared error of prediction Field 2:", mean_squared_error(f2_true,f2_predict) )
print("mean squared error of prediction Field 3:", mean_squared_error(f3_true,f3_predict) )
print("mean squared error of prediction Field 4:", mean_squared_error(f4_true,f4_predict) )