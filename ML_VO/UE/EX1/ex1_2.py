#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def slice_labels(y_s):
    """Convert soft labels to hard labels"""
    return (2*(y_s > 0))-1

class Classifier:

    def __init__(self, x_train, y_train):
        self.weights = self._train(x_train, y_train)

    def _train(self, x, y):
        n_samples = len(y)
        mat_x = np.hstack([x, np.ones((n_samples,1))])
        weights = np.linalg.pinv(mat_x) @ y
        return weights

    def classify_soft(self, x):
        """Returns the soft-labels corresponding to the argument x"""
        n_values = len(x)
        mat_x = np.hstack([x, np.ones((n_values,1))])
        return mat_x @ self.weights

    def classify_hard(self, x):
        """Returns the hard-labels corresponding to the argument x"""
        return slice_labels(self.classify_soft(x))

    def accuracy(self, x, y_truth):
        total_samples = len(x)
        errors = np.sum(np.abs(self.classify_hard(x) - y_truth) > 0)
        return (total_samples - errors)/total_samples

#%% Load dataset and plot
dataset = 1 # 0 for data_blob
            # 1 for data moon

if dataset == 0:
    data_train = pd.read_csv('data_blob_train.csv')
    data_test = pd.read_csv('data_blob_test.csv')
else:
    data_train = pd.read_csv('data_moon_train.csv')
    data_test = pd.read_csv('data_moon_test.csv')

x_train = data_train[['x_0', 'x_1']].to_numpy()
x_test = data_test[['x_0', 'x_1']].to_numpy()

y_train = data_train['y'].to_numpy()
y_test = data_test['y'].to_numpy()

# TODO
x0lim_std = [-3, 3]
x1lim_std = [-3, 3]

def plot_data(data, xlim=x0lim_std, ylim=x1lim_std):
    fig = plt.figure()
    data[data['y'] > 0].plot.scatter(x='x_0', y='x_1', color='blue',
                                     label='$y=+1$', ax=fig.gca())
    data[data['y'] < 0].plot.scatter(x='x_0', y='x_1', color='red',
                                     label='$y=-1$', ax=fig.gca()) 
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)
    return fig
    

fig_train = plot_data(data_train)
fig_test = plot_data(data_test)

#%% Train Linear classifier and determine Accuracies
my_classifier = Classifier(x_train, y_train)
accuracy_train = my_classifier.accuracy(x_train, y_train)
accuracy_test =  my_classifier.accuracy(x_test, y_test)

print('Train Accuracy = {:5.2f}%'.format(accuracy_train*100))
print('Test Accuracy = {:5.2f}%'.format(accuracy_test*100))

#%% Create Heatmap
nx0 = 100
nx1 = 100
x0_plot = np.linspace(x0lim_std[0], x0lim_std[1], nx0)
x1_plot = np.linspace(x1lim_std[0], x1lim_std[1], nx1)
xv, yv = np.meshgrid(x0_plot, x1_plot)

points = np.vstack([xv.ravel(), yv.ravel()]).T
soft_labels = my_classifier.classify_soft(points)

fig_heatmap = plt.figure()
im = plt.imshow(np.reshape(soft_labels,(nx0,nx1)), 
                cmap=plt.cm.OrRd, origin='lower',
                extent=[x0_plot.min(), x0_plot.max(), x1_plot.min(), x1_plot.max()])
cbar = fig_heatmap.colorbar(im)
cbar.set_label('$\\tilde{y}$', rotation=0)
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')

#%% Plot Decision Boundary
def decision_boundary(w, x0):
    return -1/(w[1])*(w[2]+w[0]*x0)

fig_boundary_train = plot_data(data_train)
plt.plot(x0_plot, decision_boundary(my_classifier.weights, x0_plot), c='k',
         label='Decision Boundary')
plt.legend(loc=1)

fig_boundary_test = plot_data(data_test)
plt.plot(x0_plot, decision_boundary(my_classifier.weights, x0_plot), c='k',
         label='Decision Boundary')
plt.legend(loc=1)

#%% 1.2.5 Error for two particular samples
if dataset == 0:
    s1_x = np.array([[-10], [10]]).T
    s1_y = np.array([1])
    
    s2_x = np.array([[0], [1.5]]).T
    s2_y = np.array([-1])
    
    y1_soft = my_classifier.classify_soft(s1_x)
    y2_soft = my_classifier.classify_soft(s2_x)
    
    l2err_y1 = (y1_soft - s1_y)**2
    l2err_y2 = (y2_soft - s2_y)**2
    
    print('L2-Error of soft label s1 = {}'.format(l2err_y1))
    print('L2-Error of soft label s2 = {}'.format(l2err_y2))
    
    print('L2-Error of hard label s1 = {}'.format((slice_labels(y1_soft) - s1_y)**2))
    print('L2-Error of hard label s2 = {}'.format((slice_labels(y2_soft) - s2_y)**2))

#%% Try to save figures
dataset_name = 'blob' if dataset == 0 else 'moon'
try:
    fig_train.savefig('figs/ex1_2_train_{}.pdf'.format(dataset_name), bbox_inches='tight')
    fig_test.savefig('figs/ex1_2_test_{}.pdf'.format(dataset_name), bbox_inches='tight')
    fig_heatmap.savefig('figs/ex1_2_heatm_{}.pdf'.format(dataset_name), bbox_inches='tight')
    fig_boundary_train.savefig('figs/ex1_2_b_train_{}.pdf'.format(dataset_name), bbox_inches='tight')
    fig_boundary_test.savefig('figs/ex1_2_b_test_{}.pdf'.format(dataset_name), bbox_inches='tight')
except:
    print("Error while saving figures")




