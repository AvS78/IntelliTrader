# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:28:39 2020

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

def embed_seq(
    inputs, vocab_size = None, embed_dim = None, zero_pad = False, scale = False
):
    lookup_table = tf.get_variable(
        'lookup_table', dtype = tf.float32, shape = [vocab_size, embed_dim]
    )
    if zero_pad:
        lookup_table = tf.concat(
            (tf.zeros([1, embed_dim]), lookup_table[1:, :]), axis = 0
        )
    outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    if scale:
        outputs = outputs * (embed_dim ** 0.5)
    return outputs


def learned_positional_encoding(
    inputs, embed_dim, zero_pad = False, scale = False
):
    T = inputs.get_shape().as_list()[1]
    outputs = tf.range(T)
    outputs = tf.expand_dims(outputs, 0)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])
    return embed_seq(outputs, T, embed_dim, zero_pad = zero_pad, scale = scale)


def layer_norm(inputs, epsilon = 1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims = True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable(
        'gamma', params_shape, tf.float32, tf.ones_initializer()
    )
    beta = tf.get_variable(
        'beta', params_shape, tf.float32, tf.zeros_initializer()
    )
    return gamma * normalized + beta


def pointwise_feedforward(inputs, num_units = [None, None], activation = None):
    outputs = tf.layers.conv1d(
        inputs, num_units[0], kernel_size = 1, activation = activation
    )
    outputs = tf.layers.conv1d(
        outputs, num_units[1], kernel_size = 1, activation = None
    )
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs


class Model:
    def __init__(
        self,
        dimension_input,
        dimension_output,
        seq_len,
        learning_rate,
        num_heads = 8,
        attn_windows = range(1, 6),
    ):
        self.size_layer = dimension_input
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.X = tf.placeholder(tf.float32, [None, seq_len, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        feed = self.X
        for i, win_size in enumerate(attn_windows):
            with tf.variable_scope('attn_masked_window_%d' % win_size):
                feed = self.multihead_attn(feed, self.window_mask(win_size))
        feed += learned_positional_encoding(feed, dimension_input)
        with tf.variable_scope('multihead'):
            feed = self.multihead_attn(feed, None)
        with tf.variable_scope('pointwise'):
            feed = pointwise_feedforward(
                feed,
                num_units = [4 * dimension_input, dimension_input],
                activation = tf.nn.relu,
            )
        self.logits = tf.layers.dense(feed, dimension_output)[:, -1]
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate
        ).minimize(self.cost)
        self.correct_pred = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1)
        )
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def multihead_attn(self, inputs, masks):
        T_q = T_k = inputs.get_shape().as_list()[1]
        Q_K_V = tf.layers.dense(inputs, 3 * self.size_layer, tf.nn.relu)
        Q, K, V = tf.split(Q_K_V, 3, -1)
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis = 2), axis = 0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis = 2), axis = 0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis = 2), axis = 0)
        align = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        align = align / np.sqrt(K_.get_shape().as_list()[-1])
        if masks is not None:
            paddings = tf.fill(tf.shape(align), float('-inf'))
            align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.nn.softmax(align)
        outputs = tf.matmul(align, V_)
        outputs = tf.concat(
            tf.split(outputs, self.num_heads, axis = 0), axis = 2
        )
        outputs += inputs
        return layer_norm(outputs)

    def window_mask(self, h_w):
        masks = np.zeros([self.seq_len, self.seq_len])
        for i in range(self.seq_len):
            if i < h_w:
                masks[i, : i + h_w + 1] = 1.0
            elif i > self.seq_len - h_w - 1:
                masks[i, i - h_w :] = 1.0
            else:
                masks[i, i - h_w : i + h_w + 1] = 1.0
        masks = tf.convert_to_tensor(masks)
        return tf.tile(
            tf.expand_dims(masks, 0),
            [tf.shape(self.X)[0] * self.num_heads, 1, 1],
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

sampled['index'] = sampled.index
print(sampled.head())
time_series = pd.to_datetime(sampled.iloc[:, 4]).tolist()

# we are predicting for 50 forward periods
test=sampled.iloc[3069:,0:4]
sampled=sampled.iloc[:3069,0:4]

minmax = MinMaxScaler().fit(sampled.iloc[:, 0:4].astype('float32'))
sampled_log = minmax.transform(sampled.iloc[:, 0:4].astype('float32'))
sampled_log = pd.DataFrame(sampled_log)
print(sampled_log.head())

timestamp = 20
epoch = 100
future_day = 50

tf.reset_default_graph()
modelnn = Model(
    sampled_log.shape[1],
    sampled_log.shape[1],
    timestamp,
    0.01,
    num_heads = sampled_log.shape[1],
)
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