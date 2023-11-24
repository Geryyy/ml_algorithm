#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
tf.config.set_visible_devices([], 'GPU')


def plot_metrics(history1, history2, legend1, legend2):
    val_loss1 = history1['val_loss']
    val_acc1 = history1['val_accuracy']
    loss1 = history1['loss']
    acc1 = history1['accuracy']
    
    val_loss2 = history2['val_loss']
    val_acc2 = history2['val_accuracy']
    loss2 = history2['loss']
    acc2 =  history2['accuracy']
    
    epochs1 = len(loss1)
    epochs2 = len(loss2)
    
    fig1 = plt.figure()
    fig1.suptitle('Test Set')
    plt.subplot(2,1,1)
    plt.plot(np.arange(epochs1), val_loss1, label=legend1)
    plt.plot(np.arange(epochs2), val_loss2, label=legend2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.ylim([0, 0.8])
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(np.arange(epochs1), val_acc1)
    plt.plot(np.arange(epochs2), val_acc2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.ylim([0.8, 1])
    
    fig2 = plt.figure()
    fig2.suptitle('Train Set')
    plt.subplot(2,1,1)
    plt.plot(np.arange(epochs1), loss1, label=legend1)
    plt.plot(np.arange(epochs2), loss2, label=legend2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.ylim([0, 0.8])
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(np.arange(epochs1), acc1)
    plt.plot(np.arange(epochs2), acc2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.ylim([0.8, 1])
    return fig1, fig2


def plot_metrics_housing(history):
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


#%% Models
drop_rate = 0.5
use_early_stopping = True # flag to turn on or off early stopping

baseline_model = keras.Sequential([
    layers.Input(shape=(10,), name='Input'),
    layers.Dense(128, activation='relu', name='Layer1'),
    layers.Dense(128, activation='relu', name='Layer2'),
    layers.Dense(128, activation='relu', name='Layer3'),
    layers.Dense(3, activation='softmax', name='Output')
    ])

dropout_model = keras.Sequential([
    layers.Input(shape=(10,), name='Input'),
    layers.Dense(128, activation='relu', name='Layer1'),
    layers.Dropout(drop_rate, name='Drop1'),
    layers.Dense(128, activation='relu', name='Layer2'),
    layers.Dropout(drop_rate, name='Drop2'),
    layers.Dense(128, activation='relu', name='Layer3'),
    layers.Dropout(drop_rate, name='Drop3'),
    layers.Dense(3, activation='softmax', name='Output')
    ])

baseline_model.summary()
dropout_model.summary()

#%% data

data = pd.read_csv('classification.csv')

data_x = data.iloc[:,1:11].to_numpy()
data_y = pd.get_dummies(data['y']).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, 
                                                    test_size=0.3, random_state=0)

#%% 
batch_size = 64
epochs = 200

optimizer = keras.optimizers.Adam()
loss = keras.losses.CategoricalCrossentropy()
acc = keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')
early_stop = keras.callbacks.EarlyStopping(patience=8, monitor='val_accuracy',
                                           min_delta=1E-4, mode='max')

baseline_model.compile(optimizer=optimizer, loss=loss, metrics=acc)

dropout_model.compile(optimizer=optimizer, loss=loss, metrics=acc)

hist_base = baseline_model.fit(x_train, y_train, batch_size=batch_size, 
                               epochs=epochs, validation_data=(x_test, y_test),
                               callbacks=early_stop if use_early_stopping else None)
hist_drop = dropout_model.fit(x_train, y_train, batch_size=batch_size, 
                              epochs=epochs, validation_data=(x_test, y_test))

#%%

_, accuracy_base = baseline_model.evaluate(x_test, y_test)
_, accuracy_drop = dropout_model.evaluate(x_test, y_test)

print('Final Accuracy Baseline={}'.format(accuracy_base))
print('Final Accuracy Dropout={}'.format(accuracy_drop))

#%% metrics for dropout and baseline model
fig_test, fig_train = plot_metrics(hist_base.history, hist_drop.history, 'Baseline', 'Dropout')




#%% Housing data
dataset = fetch_california_housing(as_frame=True, download_if_missing=False)
california_housing_df = dataset['frame']

features = california_housing_df.iloc[:,0:8].to_numpy()
target = california_housing_df['MedHouseVal'].to_numpy()

mu = np.mean(features, axis=0)
sigma = np.std(features, axis=0)
features_std = (features-mu)/np.tile(sigma,[len(california_housing_df),1])


x_train, x_test, y_train, y_test = train_test_split(features_std, target, 
                                                    test_size=0.2, random_state=0)

#%% Housing model

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

#%% housing model learning curve
learning_curve = plot_metrics_housing(hist.history)

print('MAE on test set = {}'.format(model.evaluate(x_test, y_test)))

#%% save png for housing model
tf.keras.utils.plot_model(model, to_file='figs/ex3_3_model.png', show_shapes=True, 
                          show_layer_names=True)

#%%
try:
    suffix = '_early' if use_early_stopping else ''
    fig_test.savefig('figs/ex3_3_test'+suffix+'.pdf', bbox_inches='tight')
    fig_train.savefig('figs/ex3_3_train'+suffix+'.pdf', bbox_inches='tight')
    learning_curve.savefig('figs/ex3_3_housing_learning_curve.pdf', bbox_inches='tight')
except:
    print("Error while saving figures")

