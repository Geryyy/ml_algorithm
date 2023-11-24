#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# supress warning of to many open plots
plt.rcParams.update({'figure.max_open_warning': 0})

class PolynomialRegressor:
    def __init__(self, x, y, degree, regularization_parameter=0,
                 use_regularization=False):
        self.degree = degree
        self.regularization_parameter = regularization_parameter
        self.weights = self._fit(x, y, use_regularization)

    def _fit(self, x, y, use_regularization):
        """Obtain weights for Polynomial Regressor through least Squares"""
        matrixX = self._construct_matrix(x)
        if use_regularization:
            rhs = matrixX.T @ y
            mat = (matrixX.T @ matrixX
                + self.regularization_parameter*np.eye(self.degree + 1))
            weights = np.linalg.solve(mat, rhs)
        else:
            weights = np.linalg.pinv(matrixX) @ y
        return weights

    def _construct_matrix(self, x):
        """Construct Vandermonde Matrix"""
        return np.vander(x, self.degree+1, increasing = True)

    def __call__(self, x):
        matrixX = self._construct_matrix(x)
        return matrixX @ self.weights

#%% 1.1.1 load data and plot datapoints
data_test = pd.read_csv('regression_test.csv')
data_train = pd.read_csv('regression_train.csv')

def plot_data():
    """Plots Train and test data and returns figure handle"""
    fig = plt.figure()
    data_train.plot.scatter(x='x', y='y', color='blue', label='Train',
                            ax=fig.gca())
    data_test.plot.scatter(x='x', y='y', color='red', label='Test',
                           ax=fig.gca())
    return fig

fig1 = plot_data()
plt.xlim([0,2])
plt.ylim([-5, 20])
plt.grid()


#%% 1.1.2 fit different degree Polynomials and calculate MSE to test data
degrees = np.arange(11)
errors = np.zeros(len(degrees))
x_plot = np.linspace(0, 2, 100) # plotting range
y_plot = np.zeros((len(degrees), len(x_plot))) # store y values for plotting

for ind, degree in enumerate(degrees):
    # Train regressor and calculate MSE to test data
    regressor = PolynomialRegressor(data_train.x, data_train.y, degree)
    y_est = regressor(data_test.x)
    errors[ind] = np.mean((y_est - data_test.y)**2)
    # y values for plotting
    y_plot[ind, :] = regressor(x_plot)
    
fig2 = plt.figure()
plt.stem(degrees, np.log10(errors), use_line_collection=True)
plt.xlabel('Polynomial Degree')
plt.ylabel('$\\log_{10}\\left(||\hat{y}-y||^2\\right)$')
plt.grid()

#%% 1.1.3 Plot Polynomial Models for different degrees
figs_deg = [] # store all figure handles

for ind,degree in enumerate(degrees):
    new_fig = plot_data()
    plt.plot(x_plot, y_plot[ind,:], label='m={}'.format(degree), color='k')
    plt.grid()
    plt.xlim([0, 2])
    plt.ylim([-5, 20])
    plt.legend()
    figs_deg.append(new_fig)

#%% 1.1.5 Ridge regression
lambda_vals = np.array([0, 0.1,1,5,10,100,500,1000])
errors_lambda = np.zeros(len(lambda_vals))
y_plot_lambda = np.zeros((len(lambda_vals), len(x_plot)))

for ind,lamb in enumerate(lambda_vals):
    #Train regressor and calculate MSE to test data
    regressor = PolynomialRegressor(data_train.x, data_train.y, 5, lamb, True)
    y_est = regressor(data_test.x)
    errors_lambda[ind] = np.mean((y_est - data_test.y)**2)
    # y values for plotting
    y_plot_lambda[ind, :] = regressor(x_plot)

# leave out lamda = 0 for plotting in stem Plot
fig3 = plt.figure()
plt.stem(np.log10(lambda_vals[1:]), np.log10(errors_lambda[1:]), 
         use_line_collection=True)
plt.xlabel('$\\log_{10}\\left(\\lambda\\right)$')
plt.ylabel('$\\log_{10}\\left(||\\hat{y}-y||^2\\right)$')
plt.grid()

#%% 1.1.7 Plot polynomial model for different values of lambda
figs_lambda = [] # store all figure handles

for ind,lamb in enumerate(lambda_vals):
    new_fig = plot_data()
    plt.plot(x_plot, y_plot_lambda[ind,:], label='$\\lambda={}$'.format(lamb),
             color='k')
    plt.grid()
    plt.xlim([0, 2])
    plt.ylim([-5, 20])
    plt.legend()
    figs_lambda.append(new_fig)
#%% Try to save figures

try:
    fig1.savefig('figs/ex1_1_data.pdf', bbox_inches='tight')
    fig2.savefig('figs/ex1_1_mse_degree.pdf', bbox_inches='tight')
    fig3.savefig('figs/ex1_1_mse_lambda.pdf', bbox_inches='tight')
    
    for ind, degree in enumerate(degrees):
        figs_deg[ind].savefig('figs/ex1_1_m{}.pdf'.format(degree), 
                              bbox_inches='tight')
    
    for ind, lamb in enumerate(lambda_vals):
        figs_lambda[ind].savefig('figs/ex1_1_lamb{}.pdf'.format(ind), 
                                 bbox_inches='tight')
except:
    print("Error while saving figures")
