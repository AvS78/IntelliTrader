# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:02:09 2020

@author: RB
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
import time
from sklearn.metrics import mean_squared_error  


def sampled_dframe(dframe, sample_code):
    mid_data=dframe['Mid']
    sampled_data=mid_data.resample(sample_code).ohlc()
    return sampled_data

class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer)

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
            tf.float32, (None, num_layers * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        rnn_W = tf.Variable(tf.random_normal((size_layer, output_size)))
        rnn_B = tf.Variable(tf.random_normal([output_size]))
        self.logits = tf.matmul(self.outputs[-1], rnn_W) + rnn_B
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
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

# create a dataframe of engineered features with same sampling frequency as sampled dataset
derived_price=pd.DataFrame(avg_price)
derived_price.columns=['avg_price']
cols=['ln_change', 'pct_change', 'ln_open_close']
derived=pd.concat([pd.Series(ln_change), pd.Series(pct_change), pd.Series(ln_open_close)], axis=1)
derived.columns=cols
derived.index=derived_price.index
derived=pd.merge(derived_price, derived, left_index=True, right_index=True)
derived.head()

print(sampled.head())
sampled['index'] = sampled.index
print(sampled.head())
time_series = pd.to_datetime(sampled.iloc[:, 4]).tolist()

minmax = MinMaxScaler().fit(sampled.iloc[:, 0:4].astype('float32'))
sampled_log = minmax.transform(sampled.iloc[:, 0:4].astype('float32'))
sampled_log = pd.DataFrame(sampled_log)
print(sampled_log.head())

def reducedimension(
    input_, dimension = 2, learning_rate = 0.01, hidden_layer = 256, epoch = 20
):

    input_size = input_.shape[1]
    X = tf.placeholder('float', [None, input_size])
    first_layer_encoder = tf.layers.dense(
        X, hidden_layer, activation = tf.nn.sigmoid
    )
    second_layer_encoder = tf.layers.dense(
        first_layer_encoder, dimension, activation = tf.nn.sigmoid
    )
    first_layer_decoder = tf.layers.dense(
        second_layer_encoder, hidden_layer, activation = tf.nn.sigmoid
    )
    second_layer_decoder = tf.layers.dense(
        first_layer_decoder, input_size, activation = tf.nn.sigmoid
    )
    cost = tf.reduce_mean(tf.square(X - second_layer_decoder))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        last_time = time.time()
        _, loss = sess.run([optimizer, cost], feed_dict = {X: input_})
        if (i + 1) % 10 == 0:
            print(
                'epoch:', i + 1, 'loss:', loss, 'time:', time.time() - last_time
            )

    vectors = sess.run(second_layer_encoder, feed_dict = {X: input_})
    tf.reset_default_graph()
    return vectors

thought_vector = reducedimension(
    sampled_log.values,
    dimension = 16,
    learning_rate = 0.001,
    hidden_layer = 128,
    epoch = 100,
)

num_layers = 1
size_layer = 128
timestamp=60
epoch = 100
dropout_rate = 0.1

tf.reset_default_graph()
modelnn = Model(0.001, num_layers, thought_vector.shape[1], size_layer, sampled_log.shape[1], dropout_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    init_value = np.zeros((1, num_layers * size_layer))
    total_loss = 0
    for k in range(0, sampled_log.shape[0] - 1, timestamp):
        index = min(k + timestamp, sampled_log.shape[0] - 1)
        batch_x = np.expand_dims(thought_vector[k:index], axis = 0)
        batch_y = sampled_log.iloc[k + 1 : index + 1].values
        last_state, _, loss = sess.run(
            [modelnn.last_state, modelnn.optimizer, modelnn.cost],
            feed_dict = {
                modelnn.X: batch_x,
                modelnn.Y: batch_y,
                modelnn.hidden_layer: init_value,
            },
        )
        loss = np.mean(loss)
        init_value = last_state
        total_loss += loss
    total_loss /= sampled_log.shape[0] // timestamp
    if (i + 1) % 10 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)
        
output_predict = np.zeros((sampled_log.shape[0] + 1, sampled_log.shape[1]))
output_predict[0] = sampled_log.iloc[0]
upper_b = (sampled_log.shape[0] // timestamp) * timestamp
init_value = np.zeros((1, num_layers * size_layer))
for k in range(0, (sampled_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(
                thought_vector[k : k + timestamp], axis = 0
            ),
            modelnn.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[k + 1 : k + timestamp + 1] = out_logits

out_logits, last_state = sess.run(
    [modelnn.logits, modelnn.last_state],
    feed_dict = {
        modelnn.X: np.expand_dims(thought_vector[upper_b:], axis = 0),
        modelnn.hidden_layer: init_value,
    },
)
init_value = last_state
output_predict[upper_b + 1 : sampled_log.shape[0] + 1] = out_logits
sampled_log.loc[sampled_log.shape[0]] = out_logits[-1]
time_series.append(time_series[-1] + timedelta(days = 1))

sampled_log = minmax.inverse_transform(output_predict)
time_series = pd.Series(time_series).dt.strftime(date_format = '%Y-%m-%d').tolist()

f1=sampled.iloc[:, 0]
f1_predict=sampled_log[:-1, 0]
f2=sampled.iloc[:, 1]
f2_predict=sampled_log[:-1, 1]
f3=sampled.iloc[:, 2]
f3_predict=sampled_log[:-1, 2]
f4=sampled.iloc[:, 3]
f4_predict=sampled_log[:-1, 3]

print("time-lag: ", timestamp)
print("mean squared error of prediction Field 1:", mean_squared_error(f1,f1_predict) )
print("mean squared error of prediction Field 2:", mean_squared_error(f2,f2_predict) )
print("mean squared error of prediction Field 3:", mean_squared_error(f3,f3_predict) )
print("mean squared error of prediction Field 4:", mean_squared_error(f4,f4_predict) )
