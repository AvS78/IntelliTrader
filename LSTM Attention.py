# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:49:45 2020

@author: RB
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import mean_squared_error 


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

class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
        attention_size = 10,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob = forget_bias
        )
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        attention_w = tf.get_variable(
            'attention_v', [attention_size], tf.float32
        )
        query = tf.layers.dense(
            tf.expand_dims(self.last_state[:, size_layer:], 1), attention_size
        )
        keys = tf.layers.dense(self.outputs, attention_size)
        align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
        align = tf.nn.tanh(align)
        self.outputs = tf.squeeze(
            tf.matmul(
                tf.transpose(self.outputs, [0, 2, 1]), tf.expand_dims(align, 2)
            ),
            2,
        )
        self.logits = tf.layers.dense(self.outputs, output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )

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

sampled['index'] = sampled.index
print(sampled.head())
time_series = pd.to_datetime(sampled.iloc[:, 4]).tolist()

minmax = MinMaxScaler().fit(sampled.iloc[:, 0:4].astype('float32'))
sampled_log = minmax.transform(sampled.iloc[:, 0:4].astype('float32'))
sampled_log = pd.DataFrame(sampled_log)
print(sampled_log.head())

num_layers = 1
size_layer = 128
timestamp = 5
epoch = 50
dropout_rate = 0.7
future_day = 50

tf.reset_default_graph()
modelnn = Model(0.01, num_layers, sampled_log.shape[1], size_layer, sampled_log.shape[1], dropout_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    init_value = np.zeros((timestamp, num_layers * 2 * size_layer))
    total_loss = 0
    for k in range(0, (sampled_log.shape[0] // timestamp) * timestamp, timestamp):
        index = min(k + timestamp, sampled_log.shape[0] - 1)
        batch_x = np.expand_dims(sampled_log.iloc[k:index].values, axis = 1)
        batch_y = sampled_log.iloc[k + 1 : index + 1].values
        last_state, _, loss = sess.run(
            [modelnn.last_state, modelnn.optimizer, modelnn.cost],
            feed_dict = {
                modelnn.X: batch_x,
                modelnn.Y: batch_y,
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        total_loss += loss
    total_loss /= sampled_log.shape[0] // timestamp
    if (i + 1) % 10 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)

output_predict = np.zeros((sampled_log.shape[0] + future_day, sampled_log.shape[1]))
output_predict[0] = sampled_log.iloc[0]
upper_b = (sampled_log.shape[0] // timestamp) * timestamp
init_value = np.zeros((timestamp, num_layers * 2 * size_layer))
for k in range(0, (sampled_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(sampled_log.iloc[k : k + timestamp], axis = 1),
            modelnn.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[k + 1 : k + timestamp + 1] = out_logits

expanded = np.expand_dims(sampled_log.iloc[upper_b:], axis = 1)
out_logits, last_state = sess.run(
    [modelnn.logits, modelnn.last_state],
    feed_dict = {
        modelnn.X: expanded,
        modelnn.hidden_layer: init_value[-expanded.shape[0] :],
    },
)
init_value[-expanded.shape[0] :] = last_state
output_predict[upper_b + 1 : sampled_log.shape[0] + 1] = out_logits
sampled_log.loc[sampled_log.shape[0]] = out_logits[-1]
time_series.append(time_series[-1] + timedelta(days = 1))

for i in range(future_day - 1):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(sampled_log.iloc[-timestamp:], axis = 1),
            modelnn.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[sampled_log.shape[0]] = out_logits[-1]
    sampled_log.loc[sampled_log.shape[0]] = out_logits[-1]
    time_series.append(time_series[-1] + timedelta(days = 1))
    
sampled_log = minmax.inverse_transform(output_predict)
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