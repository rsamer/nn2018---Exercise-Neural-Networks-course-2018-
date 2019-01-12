#!/usr/bin/env python3
#
# data_generator.py - generate data for recurrent neural network task
#
# Generate a random string from {0,1} of given length and a target label.
# The target label for string x is given by XOR(x[offset],x[offset+1])

import numpy as np

def generate_data(N, T, offset=5):
    """
    Generate N training sequences.

    args:
        N                   number of examples
        T                   sequence length
        offset              offset for target, default: 5

    returns
        X                   input sequences as array of shape (N, T)
        y                   labels as array of shape (N,)
    """

    X = np.random.randint(2, size=(N, T))
    y = X[:,offset] ^ X[:,offset+1]

    return X[...,np.newaxis], y[...,np.newaxis]
