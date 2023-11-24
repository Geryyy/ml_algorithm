#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def loss(x, y, weights):
    n_weight = weights.shape[1]
    loss = np.sum((np.tile(y,[1,n_weight])-(x@weights))**2, axis=0)
    return loss

# gradient for sgd
def grad_sample(x, y, weights):
    return 2*x[np.newaxis].T*(x@weights - y)

# gradient for bgd
def grad_avg(x, y, weights):
    N = len(y)
    inner_sum = np.sum(x*np.tile(x@weights-y,[1,2]), axis=0)
    return 2/N*inner_sum[np.newaxis].T

# Gradient Descent Base Class
class GradientDescent:
    def __init__(self, x, y, w0, alpha, niter, lossfun, gradfun):
        self.x = x
        self.y = y
        self.weights = w0
        self.lr = alpha
        self.niter = niter
        self.histw = np.zeros((w0.shape[0], niter))
        self.loss = lossfun
        self.grad = gradfun
        self.n = len(y)
        
    def _grad(self, k):
        raise NotImplementedError()
    
    def iterate(self):
        for k in range(self.niter):
            curr_grad = self._grad(k)
            self.weights = self.weights - self.lr*curr_grad
            self.histw[:,k] = self.weights[:,0]
        return self.weights

# SGD and BGD implementations of Gradient descent
class SGD(GradientDescent):
    def _grad(self, k):
        index = k % self.n
        return self.grad(self.x[index,:], self.y[index], self.weights)

class BGD(GradientDescent):
    def _grad(self, k):
        return self.grad(self.x, self.y, self.weights)

def surface_plot(errors, w0_range, w1_range):
    fig = plt.figure()
    plt.imshow(errs, origin='lower', extent=(w0_range[0], w0_range[1], 
                                             w1_range[0], w1_range[1]))
    plt.xlabel('$w_0$')
    plt.ylabel('$w_1$')
    return fig

def plots_optimizer(x, y, gd, errs, w0_range, w1_range, name):
    loss_gd = loss(x, y, gd.histw)
    fig_loss = plt.figure()
    plt.plot(np.arange(len(loss_gd))+1, loss_gd)
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.grid()
    plt.title(f'Loss {name}')
    
    fig_surf = surface_plot(errs, w0_range, w1_range)
    plt.scatter(gd.histw[0,:], gd.histw[1,:], color='red', marker='+')
    plt.title(f'{name}')
    print('Final weight vector {}: {}'.format(name, bgd.weights))
    return (fig_surf, fig_loss)

#%% read data
data = pd.read_csv('training.csv')
features = data[['x_0', 'x_1']].to_numpy()
labels = np.array(data['y'].to_numpy(), ndmin=2).T

#%%
mat_A = features.T @ features
print(f'A = {mat_A}')

#%% generate surface plot
npt = 200
w0_range = [0, 100]
w1_range = [0, 100]
w0_vals = np.linspace(w0_range[0], w0_range[1], npt)
w1_vals = np.linspace(w1_range[0], w1_range[1], npt)
w00, w11 = np.meshgrid(w0_vals, w1_vals)

weights = np.array([w00.ravel(), w11.ravel()])
errs = loss(features, labels, weights)
errs = errs.reshape((npt, npt))

fig_surf = surface_plot(errs, w0_range, w1_range)

#%%
w_init = np.array([1,1], ndmin=2).T
alpha = 1E-2
K = 200

bgd = BGD(features, labels, w_init, alpha, K, loss, grad_avg)
sgd = SGD(features, labels, w_init, alpha, K, loss, grad_sample)

bgd.iterate()
sgd.iterate()

#%% BGD plot
surf_bgd, loss_bgd = plots_optimizer(features, labels, bgd, errs, 
                                     w0_range, w1_range, 
                                     'Batch-Gradient-Descent, $\\alpha=10^{-2}$')

#%% SGD plot
surf_sgd, loss_sgd = plots_optimizer(features, labels, sgd, errs, 
                                     w0_range, w1_range, 
                                     'Stochastic-Gradient-Descent, $\\alpha=10^{-2}$')

#%% higher learning rate
bgd_hlr = BGD(features, labels, w_init, 1E-1, K, loss, grad_avg)
sgd_hlr = SGD(features, labels, w_init, 1E-1, K, loss, grad_sample)

bgd_hlr.iterate()
sgd_hlr.iterate()

surf_bgd1, loss_bgd1 = plots_optimizer(features, labels, bgd_hlr, errs, 
                                       w0_range, w1_range, 
                                       'Batch-Gradient-Descent, $\\alpha=10^{-1}$')
surf_sgd1, loss_sgd1 = plots_optimizer(features, labels, sgd_hlr, errs, 
                                       w0_range, w1_range, 
                                       'Stochastic-Gradient-Descent, $\\alpha=10^{-1}$')

#%% even higher learning rate, alpha=1
bgd_lr1 = BGD(features, labels, w_init, 1, K, loss, grad_avg)
sgd_lr1 = SGD(features, labels, w_init, 1, K, loss, grad_sample)

bgd_lr1.iterate()
sgd_lr1.iterate()

print('Final Weights with LR=1:')
print(f'BGD: {bgd_lr1.weights}')
print(f'SGD: {sgd_lr1.weights}')
print('Optimizer Diverges')

#%% save figures
try:
    fig_surf.savefig('figs/ex3_1_surf.pdf', bbox_inches='tight')
    surf_bgd.savefig('figs/ex3_1_surf_bgd.pdf', bbox_inches='tight')
    surf_sgd.savefig('figs/ex3_1_surf_sgd.pdf', bbox_inches='tight')
    surf_bgd1.savefig('figs/ex3_1_surf_bgd1.pdf', bbox_inches='tight')
    surf_sgd1.savefig('figs/ex3_1_surf_sgd1.pdf', bbox_inches='tight')
    
    loss_bgd.savefig('figs/ex3_1_loss_bgd.pdf', bbox_inches='tight')
    loss_sgd.savefig('figs/ex3_1_loss_sgd.pdf', bbox_inches='tight')
    loss_bgd1.savefig('figs/ex3_1_loss_bgd1.pdf', bbox_inches='tight')
    loss_sgd1.savefig('figs/ex3_1_loss_sgd1.pdf', bbox_inches='tight')
except:
    print("Error while saving figures")
    
        
