#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def softplus(t):
    return np.log(1+np.exp(t))

def loss_function(samples, weights, bias):
    xs, ys = samples
    n_samples = len(ys)
    loss = 0
    
    for i in range(n_samples):
        soft_label = xs[i,:] @ weights + bias
        loss += softplus(-ys[i]*soft_label)
    
    return loss/n_samples

def loss_gradient(samples, weights, bias):
    xs, ys = samples
    n_samples = len(ys)
    grad_w = 0
    grad_b = 0
    
    for i in range(n_samples):
        soft_label = xs[i,:] @ weights + bias
        factor = np.exp(-ys[i]*soft_label)/(1+np.exp(-ys[i]*soft_label))
        grad_w += factor*(-ys[i])*xs[i,:]
        grad_b += factor*(-ys[i])
    # return gradient w and gradient b as tuple
    return (grad_w/n_samples, grad_b/n_samples)

def gradient_descent(samples, init, step_size, iterations):
    weights, bias = init
    losses = np.zeros(iterations)
    parameters = np.zeros((iterations, 3))
    
    for k in range(iterations):
        grad_w, grad_b = loss_gradient(samples, weights, bias)
        # gradient descent step
        weights = weights - step_size*grad_w
        bias = bias - step_size*grad_b
        # store current loss and current parameters
        losses[k] = loss_function(samples, weights, bias)
        parameters[k] = np.hstack([weights, bias])
    return (weights, bias, parameters, losses)

def slice_labels(y_s):
    """Convert soft labels to hard labels"""
    return (2*(y_s > 0))-1

class Classifier:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def classify_soft(self, xs):
        return xs @ self.weights + self.bias
    
    def classify_hard(self, xs):
        return slice_labels(self.classify_soft(xs))
    
    def decision_boundary(self, x):
        """Decision Boundary x_1 as a function of x_0"""
        return (-self.bias-x*self.weights[0])*1/self.weights[1]
    
    def accuracy(self, samples):
        xs, y_truth = samples
        total_samples = len(y_truth)
        errors = np.sum(np.abs(self.classify_hard(xs) - y_truth) > 0)
        return (total_samples - errors)/total_samples        

#%% 2.1.1
x_range = np.linspace(-10, 10, 100)
fig1 = plt.figure()
plt.plot(x_range, softplus(x_range))
plt.xlabel('t')
plt.ylabel('$\\log(1+e^t)$')
plt.grid()
    
#%% 2.1.4
dataset = pd.read_csv('classification.csv')
samples = (dataset[['x_0','x_1']].to_numpy(), dataset['target'].to_numpy())

w0 = np.array([0,-2])
b0 = 2
K = 600
step_size = 0.01
w_end, b_end, params, losses = gradient_descent(samples, (w0,b0), step_size, K)
classifier = Classifier(w_end, b_end)
print(f'Weight vector: {w_end}')
print(f'Bias term: {b_end}')
print('Accuracy on Train set = {}%'.format(100*classifier.accuracy(samples)))

#%% 2.1.5
iterations = np.arange(K)+1
fig2 = plt.figure(figsize=(6.4,6.4))
plt.subplot(2,1,1)
plt.plot(iterations, losses)
plt.ylabel('Loss')
plt.grid()

plt.subplot(2,1,2)
plt.plot(iterations, params)
plt.xlabel('Iteration')
plt.ylabel('Parameter values')
plt.legend(['$w_0$', '$w_1$', '$b$'], loc=0)
plt.grid()

#%% 2.1.6
x_range = np.linspace(-4, 4, 100)
fig3 = plt.figure()
dataset[dataset.target > 0].plot.scatter(x='x_0', y='x_1', c='blue', 
                                         label='$y=+1$', ax=fig3.gca())
dataset[dataset.target < 0].plot.scatter(x='x_0', y='x_1', c='red', 
                                         label='$y=-1$', ax=fig3.gca())

plt.plot(x_range, classifier.decision_boundary(x_range), c='k', 
         label='Boundary')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.grid()
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.legend()

#%% 2.1.7
sample1 = (np.array([-100, 0], ndmin=2), [1])
sample2 = (np.array([-1, 0], ndmin=2), [-1])

loss1 = loss_function(sample1, w_end, b_end)
loss2 = loss_function(sample2, w_end, b_end)

print(f'Loss of sample 1 = {loss1}')
print(f'Loss of sample 2 = {loss2}')

#%% Save plots
try:
    fig1.savefig('figs/ex2_1_softplus.pdf', bbox_inches='tight')
    fig2.savefig('figs/ex2_1_grad_desc.pdf', bbox_inches='tight')
    fig3.savefig('figs/ex2_1_boundary.pdf', bbox_inches='tight')
except:
    print("Error while saving figures")

