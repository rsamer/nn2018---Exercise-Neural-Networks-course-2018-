#!/usr/bin/env python3

# Classify delayed XOR in a {0,1} string, in input strings of VARIABLE length

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

from data_generator import generate_data

tf.reset_default_graph()  # for iPython convenience

# ----------------------------------------------------------------------
# parameters

sequence_length = 20
num_train, num_valid, num_test = 2000, 500, 500

#cell_type = 'simple'
#cell_type = 'gru'
cell_type = 'lstm'
num_hidden = 20

batch_size = 40
learning_rate = 0.01
max_epoch = 200

# ----------------------------------------------------------------------

# Generate delayed XOR samples
X_train, y_train = generate_data(num_train, sequence_length)
sl_train = sequence_length * np.ones(num_train) # NEW

X_valid, y_valid = generate_data(num_valid, sequence_length)
sl_valid = sequence_length * np.ones(num_valid) # NEW

X_test, y_test = generate_data(num_test, sequence_length)
sl_test = sequence_length * np.ones(num_test) # NEW

# Crop data
# Artificially define variable sequence lengths
# for demo-purposes
for i in range(num_train):
    ll = 10+random.randint(0,sequence_length-10)
    sl_train[i] = ll

for i in range(num_valid):
    ll = 10+random.randint(0,sequence_length-10)
    sl_valid[i] = ll

for i in range(num_test):
    ll = 10+random.randint(0,sequence_length-10)
    sl_test[i] = ll

# placeholder for the sequence length of the examples
seq_length = tf.placeholder(tf.int32, [None])

# input tensor shape: number of examples, input length, dimensionality of each input
# at every time step, one bit is shown to the network
X = tf.placeholder(tf.float32, [None, sequence_length, 1])

# output tensor shape: number of examples, dimensionality of each output
# Binary output at end of sequence
y = tf.placeholder(tf.float32, [None, 1])

# define recurrent layer
if cell_type == 'simple':
  cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
  # cell = tf.keras.layers.SimpleRNNCell(num_hidden) #alternative
elif cell_type == 'lstm':
  cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
elif cell_type == 'gru':
  cell = tf.nn.rnn_cell.GRUCell(num_hidden)
else:
  raise ValueError('bad cell type.')
# Cells are one fully connected recurrent layer with num_hidden neurons
# Activation function can be defined as second argument.
# Standard activation function is tanh for BasicRNN and GRU


# only use outputs, ignore states
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=seq_length) # NEW
# tf.nn.dynamic_rnn(cell, inputs, ...)
# Creates a recurrent neural network specified by RNNCell cell.
# Performs fully dynamic unrolling of inputs.
# Returns:
# outputs: The RNN output Tensor shaped: [batch_size, max_time, cell.output_size].

# get the unit outputs at the last time step
last_outputs = outputs[:,-1,:]

# add output neuron
y_dim = int(y.shape[1])
w = tf.Variable(tf.truncated_normal([num_hidden, y_dim]))
b = tf.Variable(tf.constant(.1, shape=[y_dim]))

y_pred = tf.nn.xw_plus_b(last_outputs, w, b)
# Matrix multiplication with bias

# define loss, minimizer and error
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

mistakes = tf.not_equal(y, tf.maximum(tf.sign(y_pred), 0))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# split data into batches

num_batches = int(X_train.shape[0] / batch_size)
X_train_batches = np.array_split(X_train, num_batches)
y_train_batches = np.array_split(y_train, num_batches)
sl_train_batches = np.array_split(sl_train, num_batches)

# train

error_train_ = []
error_valid_ = []

for n in range(max_epoch):
    print('training epoch {0:d}'.format(n+1))

    for X_train_cur, y_train_cur, sl_train_cur in zip(X_train_batches, y_train_batches, sl_train_batches):
        sess.run(train_step, feed_dict={X: X_train_cur, y: y_train_cur, seq_length: sl_train_cur})
        # We also need to feed the current sequence length
    error_train = sess.run(error, {X: X_train, y: y_train, seq_length: sl_train})
    error_valid = sess.run(error, {X: X_valid, y: y_valid, seq_length: sl_valid})

    print('  train:{0:.3g}, valid:{1:.3g}'.format(error_train, error_valid))

    error_train_ += [error_train]
    error_valid_ += [error_valid]

    if error_train == 0:
        break

error_test = sess.run(error, {X: X_test, y: y_test, seq_length: sl_test})
print('-'*70)
print('test error after epoch {0:d}: {1:.3f}'.format(n+1, error_test))

sess.close()

plt.figure()

plt.plot(np.arange(n+1), error_train_, label='training error')
plt.plot(np.arange(n+1), error_valid_, label='validation error')
plt.axhline(y=error_test, c='C2', linestyle='--', label='test error')
plt.xlabel('epoch')
plt.xlim(0, n)
plt.legend(loc='best')
plt.tight_layout()

plt.show()

