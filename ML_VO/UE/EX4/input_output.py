import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Conv2D, Input
from tensorflow.keras import Model
tf.config.set_visible_devices([], 'GPU')

'''
Examine the following toy examples to obtain the input output relations for
the configurations defined in task 4.1.1. Note that the input to a CONV1D layer
has the shape (batch_size, series_length, channels)
'''

'''
One-Channel-Input
'''

X = np.array([[
    [1],
    [3],
    [5],
    [7]
]])

weights = np.array([
    [ [1, 1, 1] ],
    [ [2, 2, 3] ],
])
bias = np.array([4, 4, 0])

input = Input(shape=(4, 1))
output = Conv1D(3, 2, padding="valid", activation="sigmoid")(input)
model = Model(input, output)

model.layers[1].set_weights([weights, bias])
model.layers[1].get_weights()
model.predict(X)
#%%
'''
Two-Channel-Input
'''

X = np.array([[
    [1, 2],
    [-1, 1],
    [5, 6],
    [7, 8]
]])

weights = np.array([
    [ [1], [-2] ],
    [ [1], [-1] ]
])
bias = np.array([0])

input = Input(shape=(4, 2))
output = Conv1D(1, 2,strides=1,padding="valid", activation="sigmoid")(input)
model = Model(input, output)

model.layers[1].set_weights([weights, bias])
model.layers[1].get_weights()
model.predict(X)

#%%

'''
2D-Convolutional Layer
'''

X = np.array([[1,2,3,4],
              [2,3,4,5],
              [3,4,5,6],
              [4,5,6,7]])[np.newaxis,...,np.newaxis]


input = Input(shape=(4, 4, 1))
output = Conv2D(1, (2,2), padding="valid", activation="sigmoid")(input)
model = Model(input, output)

model.layers[1].get_weights()

weights = np.array([[[1], [1]], [[1], [1]]])[...,np.newaxis]
bias = np.array([0])
model.layers[1].set_weights([weights, bias])
