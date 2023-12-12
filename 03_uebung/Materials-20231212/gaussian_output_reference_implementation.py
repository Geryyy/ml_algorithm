'''
- Below you can find one possible implementation of the negative log-likelihood loss which can be passed to model.compile().
- This implementation requires the output to be a concatenation of mu and sigma estimates, which should stem from your own model.
- Note, that the loss function is implemented with tf.math functions, such that the gradients of the overall model are available.
'''

import tensorflow as tf
import numpy as np

def negloglik(y_true, y_pred):
    mean, sigma = tf.reshape(y_pred[:, 0],(-1, 1)), tf.reshape(y_pred[:, 1],(-1, 1)) #Split the mean and sigma output
    loss = 0.5 * np.log(2. * np.pi) + tf.math.log(sigma) + 0.5 * (y_true / sigma - mean / sigma)**2 #Compute the negloglik loss
    return loss


#The neural network generates a intermediate output from which mu and sigma are computed
input = tf.keras.layers.Input(shape=(....), name="input")
x = tf.keras.layers.Dense(...)(input)
intermediate_output = tf.keras.layers.Dense(....)(x)

mu = tf.keras.layers.Dense(1, activation="linear")(intermediate_output)
scale_raw = tf.keras.layers.Dense(1, activation="linear")(intermediate_output)

#The following activation maps sigma to a valid range and behaves well numerically
sigma = tf.keras.layers.Lambda(lambda t:  1e-3 + tf.math.softplus(0.05 * t))(sigma_raw)

#The loss function implementation requires the output to be a concatenation of mu and sigma estimates
output = tf.keras.layers.Concatenate(axis=1)([mu, sigma], name="output")
