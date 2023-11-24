#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def plot_data(data, xlim, ylim, xind, yind):
    fig = plt.figure()
    data[data['y'] > 0.5].plot.scatter(xind, yind, color='blue',
                                       label='$y=1$', ax=fig.gca())
    data[data['y'] < 0.5].plot.scatter(xind, yind, color='red',
                                       label='$y=0$', ax=fig.gca()) 
    plt.xlabel('$'+xind+'$')
    plt.ylabel('$'+yind+'$')
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)
    return fig

def decision_boundary(x, weights, bias):
    return (-bias-x*weights[0])*1/weights[1]

def transformation(xs):
    x2 = xs[:,0]**2+xs[:,1]**2
    return np.column_stack((xs, x2))

# select dataset:
# 0 blobs dataset
# 1 circles dataset
selector = 1

#%% read dataset
if selector == 0:
    dataset = pd.read_csv('blobs.csv')
    x_min = 0
    y_min = 0
    x_max = 4.5
    y_max = 4.5
else:
    dataset = pd.read_csv('circles.csv')
    x_min = -2
    y_min = -2
    x_max = 2
    y_max = 2

x_train = dataset[['x_0', 'x_1']].to_numpy()
y_train = dataset['y'].to_numpy()

#%% Linear SVC
svc_lin = SVC(C=1, kernel='linear')
svc_lin.fit(x_train, y_train)

weights = svc_lin.coef_[0]
bias = svc_lin.intercept_

print(f'SVC weights = {weights}')
print(f'SVC bias    = {bias}')

# plot dataset with decision surface
x_range = np.linspace(x_min, x_max, 100)
fig1 = plot_data(dataset, [x_min, x_max], [y_min, y_max], 'x_0', 'x_1')
plt.plot(x_range, decision_boundary(x_range, weights, bias), c='k', 
         label='$\mathbf{x}^T\mathbf{w}+b=0$')
if selector == 0:
    plt.plot(x_range, decision_boundary(x_range, weights, bias-1), 
             c='lightskyblue', label='$\mathbf{x}^T\mathbf{w}+b=+1$')
    plt.plot(x_range, decision_boundary(x_range, weights, bias+1), 
             c='lightcoral', label='$\mathbf{x}^T\mathbf{w}+b=-1$')
plt.legend(loc=0)

if selector == 0:
    print('Margin = {}'.format(2/np.linalg.norm(weights)))

#%% for circles dataset
if selector == 1:
    #%% transform to extend dataset
    x_train_extended = transformation(x_train)
    dataset_ext = pd.DataFrame(data=x_train_extended, columns=['x_0', 'x_1', 'x_2'])
    dataset_ext['y'] = dataset.y
    
    svc_ex = SVC(C=100, kernel='linear')
    svc_ex.fit(x_train_extended, y_train)
    weights_ex = svc_ex.coef_[0]
    bias_ex = svc_ex.intercept_[0]

    fig_ext = plot_data(dataset_ext, [x_min, x_max], [0, 1.5], 'x_0', 'x_2')
    plt.plot(x_range, decision_boundary(x_range, weights_ex[[0,2]], bias_ex),
             c='k', label='$\mathbf{x}^T\mathbf{w}+b=0$')
    plt.plot(x_range, decision_boundary(x_range, weights_ex[[0,2]], bias_ex+1),
             c='lightcoral', label='$\mathbf{x}^T\mathbf{w}+b=-1$')
    plt.plot(x_range, decision_boundary(x_range, weights_ex[[0,2]], bias_ex-1),
             c='lightskyblue', label='$\mathbf{x}^T\mathbf{w}+b=+1$')
    plt.legend(loc=1)
    
    #%% SVC with polynomial kernel
    svc_poly = SVC(C=1, kernel='poly', degree=2)
    svc_poly.fit(x_train, y_train)
    
    nx0 = 100
    nx1 = 100
    x0_plot = np.linspace(x_min, x_max, nx0)
    x1_plot = np.linspace(y_min, y_max, nx1)
    xv, yv = np.meshgrid(x0_plot, x1_plot)

    points = np.vstack([xv.ravel(), yv.ravel()]).T
    soft_labels = svc_poly.predict(points)
    
    fig_heatmap = plt.figure()
    im = plt.imshow(np.reshape(soft_labels,(nx0,nx1)), 
                    cmap=plt.cm.OrRd, origin='lower',
                    extent=[x0_plot.min(), x0_plot.max(), x1_plot.min(), x1_plot.max()])
    cbar = fig_heatmap.colorbar(im)
    cbar.set_label('$\\hat{y}$', rotation=0)
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    
#%% save figures
try:
    if selector == 0:
        fig1.savefig('figs/ex2_2_blob_boundary.pdf', bbox_inches='tight')
    else:
        fig1.savefig('figs/ex2_2_circ_boundary.pdf', bbox_inches='tight')
        fig_ext.savefig('figs/ex2_2_circ_ext.pdf', bbox_inches='tight')
        fig_heatmap.savefig('figs/ex2_2_circ_heatmap.pdf', bbox_inches='tight')
except:
    print("Error while saving figures")
