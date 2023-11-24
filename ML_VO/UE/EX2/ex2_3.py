#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Martin Baumann, 01527563
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def print_metrics(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    tn = conf_mat[0,0]
    fp = conf_mat[0,1]
    fn = conf_mat[1,0]
    tp = conf_mat[1,1]
    
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp) if tp+fp > 0 else np.nan
    recall = tp/(tp+fn) if tp+fn > 0 else np.nan
    if (precision > 0 and recall > 0 and not np.isnan(precision)
        and not np.isnan(recall)):
        f1_score = 2/(1/precision+1/recall)
    else:
        f1_score = np.nan
    print('Confusion Matrix:')
    print(conf_mat)
    print(f'Acurracy = {accuracy}')
    print(f'Precision = {precision}')
    print(f'Recall = {recall}')
    print(f'F1-Score={f1_score}')
    print('\n')

def extract_xy(dataset):
    return (dataset[['x_0', 'x_1']], dataset['y'])

data_train = pd.read_csv('imbalanced_train.csv')
data_test  = pd.read_csv('imbalanced_test.csv')

x_train, y_train = extract_xy(data_train)
x_test, y_test = extract_xy(data_test)

ind_train_maj = data_train.y < 0.5
ind_train_min = data_train.y > 0.5

ind_test_maj = data_test.y < 0.5
ind_test_min = data_test.y > 0.5

#%% histogram
fig_hist = plt.figure(figsize=(8,4.8))
ax1 = plt.subplot(1,2,1)
data_train.hist(column='y', ax=ax1)
plt.xlabel('y')
plt.ylabel('Frequency')
plt.title('Train')

ax2 = plt.subplot(1,2,2)
data_test.hist(column='y', ax=ax2)
plt.xlabel('y')
plt.ylabel('Frequency')
plt.title('Test')

#%% full dataset
svc_full = SVC(C=0.01, kernel='rbf')
svc_full.fit(x_train, y_train)

print('### Full Dataset ###')
print_metrics(y_test, svc_full.predict(x_test))

#%% undersampling
n_minority = sum(data_train.y)
n_majority = len(data_train) - n_minority

# sample from majority class
data_under = data_train[ind_train_maj].sample(n=n_minority, random_state=1)
# add all minority class elements
data_under = data_under.append(data_train[ind_train_min])
# shuffle dataset
data_under = data_under.sample(frac=1, random_state=1)

x_train_under, y_train_under = extract_xy(data_under)

svc_under = SVC(C=0.01, kernel='rbf')
svc_under.fit(x_train_under, y_train_under)

print('### Undersampling ###')
print_metrics(y_test, svc_under.predict(x_test))

#%% oversampling
# add all majority class elements
data_over = data_train[ind_train_maj]
# add samples from minority class
data_over = data_over.append(data_train[ind_train_min].
                             sample(n=n_majority, random_state=1, replace=True))
# shuffle dataset
data_over = data_over.sample(frac=1, random_state=1)

x_train_over, y_train_over = extract_xy(data_over)

svc_over = SVC(C=0.01, kernel='rbf')
svc_over.fit(x_train_over, y_train_over)

print('### Oversampling ###')
print_metrics(y_test, svc_over.predict(x_test))

#%% save figures

try:
    fig_hist.savefig('figs/ex2_3_hist.pdf', bbox_inches='tight')
except:
    print("Error while saving figures")
