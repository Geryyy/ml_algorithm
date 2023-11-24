#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
tf.config.set_visible_devices([], 'GPU')

im_selector = 1 # 0 ... edges
                # 1 ... bike

#%% read images
im_edges = tf.image.rgb_to_grayscale(mpimg.imread('edges.png'))[np.newaxis,...]
im_bike = tf.image.rgb_to_grayscale(mpimg.imread('bike.png'))[np.newaxis,...]

shape_edges = im_edges.shape
shape_bike = im_bike.shape

#%% setup filters
filt1 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])

filt2 = filt1.T

filt1 = filt1[...,np.newaxis]
filt2 = filt2[...,np.newaxis]

filters_stack = np.stack([filt1, filt2], axis=3)
biases = np.array([0, 0])

#%% setup keras model
if im_selector == 0:
    input_shape = tuple([*shape_edges[1:3], 1])
    img = im_edges
else:
    img = im_bike
    input_shape = tuple([*shape_bike[1:3], 1])

input_layer = keras.layers.Input(shape=input_shape)
conv2d_layer = keras.layers.Conv2D(2, 3, padding='valid', activation='linear')(input_layer)

model = keras.Model(input_layer, conv2d_layer)

model.layers[1].set_weights([filters_stack, biases])

im_filt = model(img)
    
im_filt1 = im_filt[0,:,:,0]
im_filt2 = im_filt[0,:,:,1]

#%% plot results
fig_img = plt.figure()
plt.imshow(img[0,:,:,0], cmap=plt.cm.gray)
plt.gca().set_axis_off()

fig_filt, axes = plt.subplots(1, 2, figsize=(8,4))
axes[0].set_axis_off()
axes[1].set_axis_off()

im1 = axes[0].imshow(im_filt1, cmap=plt.cm.gray)
fig_filt.colorbar(im1, ax=axes[0], shrink=0.8, fraction=0.1)
axes[0].set_title('$W_{(0)}$')

im2 = axes[1].imshow(im_filt2, cmap=plt.cm.gray)
fig_filt.colorbar(im2, ax=axes[1], shrink=0.8, fraction=0.1)
axes[1].set_title('$W_{(1)}$')

#%% save plots
try:
    suffix = 'edges' if im_selector == 0 else 'bike'
    fig_filt.savefig(f'figs/ex4_1_filt_{suffix}.pdf', bbox_inches='tight')
except:
    print('Error saving figs')    
