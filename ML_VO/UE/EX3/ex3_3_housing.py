#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
tf.config.set_visible_devices([], 'GPU')

def plot_metrics(history):
    val_loss = history['val_loss']
    loss = history['loss']
    
    epochs = len(loss)
    
    fig = plt.figure()
    plt.plot(np.arange(epochs), val_loss, label='Test')
    plt.plot(np.arange(epochs), loss, label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid()
    plt.ylim([0, 0.8])
    plt.legend()
    return fig


#%% data
dataset = fetch_california_housing(as_frame=True, download_if_missing=False)
california_housing_df = dataset['frame']

features = california_housing_df.iloc[:,0:8].to_numpy()
target = california_housing_df['MedHouseVal'].to_numpy()

mu = np.mean(features, axis=0)
sigma = np.std(features, axis=0)
features_std = (features-mu)/np.tile(sigma,[len(california_housing_df),1])


x_train, x_test, y_train, y_test = train_test_split(features_std, target, 
                                                    test_size=0.2, random_state=0)

#%% model

drop_rate = 0.15
batch_size = 64
epochs = 200
optimizer = keras.optimizers.Adam()
loss = keras.losses.MeanAbsoluteError(name='mae')

model = keras.Sequential([
    keras.layers.Input(shape=(8,), name='Input'),
    keras.layers.Dense(128, activation='relu', name='Layer1'),
    keras.layers.Dropout(drop_rate, name='Drop1'),
    keras.layers.Dense(128, activation='relu', name='Layer2'),
    keras.layers.Dropout(drop_rate, name='Drop2'),
    keras.layers.Dense(64, activation='relu', name='Layer3'),
    keras.layers.Dropout(drop_rate, name='Drop3'),
    keras.layers.Dense(1, activation='linear', name='Output')
    ])

model.summary()

model.compile(optimizer=optimizer, loss=loss)

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                 validation_data=(x_test,y_test))

#%%
learning_curve = plot_metrics(hist.history)

print('MAE on test set = {}'.format(model.evaluate(x_test, y_test)))

#%%
tf.keras.utils.plot_model(model, to_file='figs/ex3_3_model.png', show_shapes=True, 
                          show_layer_names=True)

try:
    learning_curve.savefig('figs/ex3_3_housing_learning_curve.pdf', bbox_inches='tight')
except:
    print("Error while saving figures")