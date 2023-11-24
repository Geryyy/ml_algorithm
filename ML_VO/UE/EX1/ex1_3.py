#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import sklearn.metrics as metrics
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#%% 1.3.1 Load dataset and perform sanity checks
dataset = fetch_california_housing(as_frame=True, download_if_missing=False)

california_housing_df = dataset['frame']
print(california_housing_df.describe())

print('### Correlation with MedHouseVal ###')
print(california_housing_df.corr()['MedHouseVal'].sort_values(ascending=False))


#%% Scatter plot and histogram
california_housing_df.plot.scatter(x='Longitude', y='Latitude',
                                   c='MedHouseVal', colormap='viridis')
fig1 = plt.gcf()
plt.ylabel('Longitude')

california_housing_df.hist(column='Longitude', bins=50)
fig_hist_lon = plt.gcf()
plt.title('')
plt.ylabel('Frequency')
plt.xlabel('Longitude')
california_housing_df.hist(column='Latitude', bins=50)
fig_hist_lat = plt.gcf()
plt.title('')
plt.ylabel('Frequency')
plt.xlabel('Latitude')

#%% Split dataset into train and test set

# shuffle dataset
california_housing_df = california_housing_df.sample(frac=1, random_state=1)

data_train = california_housing_df.iloc[:15000]
data_test = california_housing_df.iloc[15000:]

y_full = california_housing_df['MedHouseVal']
x_full = california_housing_df.drop('MedHouseVal', axis=1)
y_train = data_train['MedHouseVal']
x_train = data_train.drop('MedHouseVal', axis=1)
y_test  = data_test['MedHouseVal']
x_test  = data_test.drop('MedHouseVal', axis=1)

#%% Ridge regressor and RandomForestRegressor
def regressor_characterise(regressor, name):
    #Train regressor
    regressor.fit(x_train, y_train)

    # Errors of regressor
    y_pred_train = regressor.predict(x_train)
    y_pred_test = regressor.predict(x_test)
    
    mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
    mae_test = metrics.mean_absolute_error(y_test,  y_pred_test)
    err_train = (y_pred_train-y_train)
    err_test = (y_pred_test-y_test)

    print('MAE Train {}-Regressor = {}'.format(name, mae_train))
    print('MAE Test {}-Regressor = {}'.format(name, mae_test))
    print('Bias on test set = {}'.format(np.mean(err_test)))

    # Scatter Plot    
    fig_scatter = plt.figure()
    plt.scatter(y_test, y_pred_test)
    plt.xlabel('$y$')
    plt.ylabel('$\hat{y}$')
    plt.xlim([0, 5.5])
    plt.ylim([0, 5.5])
    plt.grid()
    
    # Histograms
    fig_hist1 = plt.figure()
    plt.hist(err_train, bins=50)
    plt.xlabel('Train error $(\hat{y} - y)$')
    plt.ylabel('Frequency')
    plt.xlim([-4,4])
    fig_hist2 = plt.figure()
    plt.hist(err_test, bins=50)
    plt.xlabel('Test error $(\hat{y} - y)$')
    plt.ylabel('Frequency')
    plt.xlim([-4,4])
    return [fig_scatter, fig_hist1, fig_hist2]


linear_regressor = Ridge(alpha=1)
rand_forest_regressor = RandomForestRegressor(n_estimators=50)

figs_lin = regressor_characterise(linear_regressor, 'Ridge')
figs_rf = regressor_characterise(rand_forest_regressor, 'Random Forest')

#%% Coefficiennts and Intercept of linear Regressor
coefs = pd.DataFrame(linear_regressor.coef_, 
                     index=data_train.columns.drop('MedHouseVal'))
intercept = linear_regressor.intercept_

print('### Coefficients of Linear Regressor ###')
print(coefs.sort_values(by=0, ascending=False))
print('Intercept = {}'.format(intercept))
#%% Cross Validation

score = cross_val_score(rand_forest_regressor, x_full, y_full, cv=10)
print("Mean: {:.3f}".format(score.mean()))
print("Std deviation: {:.4f}".format(score.std()))

#%% save figures

try:
    fig1.savefig('figs/ex1_3_lon_lat.pdf', bbox_inches='tight')
    fig_hist_lon.savefig('figs/ex1_3_hist_lon.pdf', bbox_inches='tight')
    fig_hist_lat.savefig('figs/ex1_3_hist_lat.pdf', bbox_inches='tight')
    figs_lin[0].savefig('figs/ex1_3_lin_scatter.pdf', bbox_inches='tight')
    figs_lin[1].savefig('figs/ex1_3_lin_hist_train.pdf', bbox_inches='tight')
    figs_lin[2].savefig('figs/ex1_3_lin_hist_test.pdf', bbox_inches='tight')
    figs_rf[0].savefig('figs/ex1_3_rf_scatter.pdf', bbox_inches='tight')
    figs_rf[1].savefig('figs/ex1_3_rf_hist_train.pdf', bbox_inches='tight')
    figs_rf[2].savefig('figs/ex1_3_rf_hist_test.pdf', bbox_inches='tight')
    
except:
    print("Error while saving figures")


