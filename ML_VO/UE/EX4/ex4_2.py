#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""


from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
tf.config.set_visible_devices([], 'GPU')

def plot_metrics(history):
    val_loss = history['val_loss']
    loss = history['loss']
    acc = history['Accuracy']
    val_acc = history['val_Accuracy']
    
    epochs = len(loss)
    
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.arange(epochs), val_loss, label='Test')
    plt.plot(np.arange(epochs), loss, label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.ylim([0, 0.3])
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(np.arange(epochs), val_acc, label='Test')
    plt.plot(np.arange(epochs), acc, label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.ylim([0.9, 1])
    plt.legend()
    
    return fig


#%% load and preprocess data

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalize x to [0, 1]
x_train_norm = x_train[..., np.newaxis]/255.
x_test_norm = x_test[..., np.newaxis]/255.

# one hot encoding of y
y_train_oh = keras.utils.to_categorical(y_train)
y_test_oh = keras.utils.to_categorical(y_test)

# shapes of dataset
n_samples_train, size_x, size_y = x_train.shape
n_samples_test = x_test.shape[0]

#%% plot samples
labels = np.unique(y_train)

fig_digits = plt.figure(figsize=(8,9))

for i in range(10):
    if i == 9:
        ax = plt.subplot(4,3,11)
    else:
        ax = plt.subplot(4,3,i+1)
    plt.imshow(x_train[np.where(y_train==labels[i])[0][0]], cmap=plt.cm.gray_r)
    ax.set_axis_off()
    plt.title(f'$y={i}$')

plt.show()

#%% SVC
# flatten the images to a vector
x_train_flat = np.reshape(x_train_norm, [n_samples_train, size_x*size_y])
x_test_flat = np.reshape(x_test_norm, [n_samples_test, size_x*size_y])

svc = SVC(C=1, kernel='rbf')
print('Fitting SVC ... ')
svc.fit(x_train_flat, y_train)

print('SVC Accuracy on Test Set = ')
print(svc.score(x_test_flat, y_test))

#%% conv nn

batch_size = 64
epochs = 20
drop_rate = 0.5
optimizer = keras.optimizers.Adam()

# due to the linear output layer from_logits has to be True
loss = keras.losses.CategoricalCrossentropy(from_logits=True, name='CrossEnt')
acc = keras.metrics.CategoricalAccuracy(name='Accuracy')

model = keras.Sequential([
    keras.layers.Input(shape=(28,28,1,), name='Input'),
    keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(2,2), activation='relu', padding='valid', name='Conv1'),
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), activation='relu', padding='valid', name='Conv2'),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid', name='Conv3'),
    keras.layers.Flatten(name='Flatten'),
    keras.layers.Dropout(drop_rate),
    keras.layers.Dense(256, activation='relu', name='Dense1'),
    keras.layers.Dropout(drop_rate),
    keras.layers.Dense(10, activation='linear', name='Output'),
    ])

model.summary()

#%% Train Model
model.compile(optimizer=optimizer, loss=loss, metrics=acc)

history = model.fit(x_train_norm, y_train_oh, batch_size=batch_size, 
                    epochs=epochs, validation_data=(x_test_norm,y_test_oh))

#%% Plot learning curve
learning_curve = plot_metrics(history.history)
print('NN Accuracy on Test Set:')
print(model.evaluate(x_test_norm, y_test_oh)[1])

#%% Confusion matrix:
y_pred_oh = model.predict(x_test_norm)
y_pred = np.argmax(y_pred_oh, axis=1) # go back from 1 hot encoding to integer labels
conf_mat = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix: \n {conf_mat}')

#%% incorrectly classified images
err_pos = y_pred != y_test
ind_err = np.argwhere(err_pos) # indices of the samples that where incorrectly classified

fig_incorr = plt.figure(figsize=(8,9))

for i in range(10):
    if i == 9:
        ax = plt.subplot(4,3,11)
    else:
        ax = plt.subplot(4,3,i+1)
    
    current_index = ind_err[i][0]
    predicted_label = y_pred[current_index]
    true_label = y_test[current_index]
    
    plt.imshow(x_test[current_index,...], cmap=plt.cm.gray_r)
    ax.set_axis_off()
    plt.title(f'$\\hat{{y}}={predicted_label}$, $y={true_label}$')

#%% plot response of conv layers to some samples
ind = 1705 # index of random sample

fig_inp = plt.figure()
plt.imshow(x_test[ind,...], cmap=plt.cm.gray_r)
plt.gca().set_axis_off()

layers = model.layers
outconv1 = layers[0](x_test_norm[[ind],...])
outconv2 = layers[1](outconv1)
outconv3 = layers[2](outconv2)
flattened = layers[3](outconv3)
dense1 = layers[5](flattened)
output = layers[7](dense1)
probs = tf.nn.softmax(output) # pass through softmax to obtain probabilities

fig_features1, axes1 = plt.subplots(1, 5, figsize=(12, 3))
fig_features2, axes2 = plt.subplots(1, 5, figsize=(12, 3))

for i in range(5):
    feat1 = axes1[i].imshow(outconv1[0,...,i])
    axes1[i].set_axis_off()
    fig_features1.colorbar(feat1, ax=axes1[i], shrink=0.6)
    feat2 = axes2[i].imshow(outconv3[0,...,i])
    axes2[i].set_axis_off()
    fig_features2.colorbar(feat2, ax=axes2[i], shrink=0.6)

#%% save plots
try:
    fig_digits.savefig('figs/ex4_2_digits.pdf', bbox_inches='tight')
    learning_curve.savefig('figs/ex4_2_learning_curve.pdf', bbox_inches='tight')
    keras.utils.plot_model(model, to_file='figs/ex4_2_model.png', show_shapes=True, 
                          show_layer_names=True)
    fig_incorr.savefig('figs/ex4_2_incorrect.pdf', bbox_inches='tight')
    fig_inp.savefig('figs/ex4_2_inp.pdf', bbox_inches='tight')
    fig_features1.savefig('figs/ex4_2_features1.pdf', bbox_inches='tight')
    fig_features2.savefig('figs/ex4_2_features2.pdf', bbox_inches='tight')
except:
    print('Error saving figs')
