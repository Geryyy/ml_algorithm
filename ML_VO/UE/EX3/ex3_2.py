#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# I had to add this line, otherwise the training wouldn't start:
tf.config.set_visible_devices([], 'GPU')

def print_model_output(model, name):
    xs = np.array([[0,0], [0,1], [1,0], [1,1]])
    print(f'Model {name}:')
    for i in range(xs.shape[0]):
        x = xs[np.newaxis,i,:]
        val = model(x)
        print(f'Model({x}) = {val}')
    print('\n')

#%% XOR

model_xor = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(2, activation='relu', name='Layer0'),
    layers.Dense(1, activation='linear', name='Layer1')])

model_xor.set_weights([
    np.array([[-1, 2],[2, -1]]), 
    np.array([-1, -1]),
    np.array([[1], [1]]),
    np.array([0])])

#%% Train Data for AND
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = 1.*np.logical_and(x[:,0], x[:,1])

#%% AND Model
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(2, activation='linear', name='Layer0'),
    layers.Dense(1, activation='sigmoid', name='Layer1')
    ])

model.summary()

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.8),
              loss=keras.losses.BinaryCrossentropy())

model.fit(x, y, batch_size=1, epochs=100)

#%% Outputs of models
print_model_output(model_xor, 'XOR')
print_model_output(model, 'AND')

#%% weights and bias
print(model.trainable_variables)
