#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:31:40 2020

@author: martin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

class KMeans:

    def __init__(self, n_clusters, init='random'):
        self.k = n_clusters
        self.init = init

        if init == 'random':
            self.initmethod = self._random_init
        elif init == 'k-means++':
            self.initmethod = self._kmpp_init
        else:
            raise ValueError(f'Init method "{init}" not supported')

        self.centroids = []

    def _random_init(self, X):
        n_samples = X.shape[0]
        indices = np.random.choice(np.arange(n_samples), size=self.k, replace=False)
        return X[indices,:]

    def _kmpp_init(self, X):
        centroids = np.zeros((self.k, X.shape[1]))
        n_samples = X.shape[0]

        # select 1st index randomly
        new_ind = np.random.randint(0, n_samples)
        centroids[0,:] = X[new_ind,:]
        
        # no need to remove already taken ones, their prob will be 0 anyway
        sample_inds = np.arange(0, n_samples)

        for i in range(1, self.k):
            _, min_dist = self._nearest_centroids(X, centroids[:i,:])
            # probability of assignment
            probs = min_dist**2/(np.sum(min_dist**2))
            new_ind = np.random.choice(sample_inds, size=1, p=probs)
            centroids[i,:] = X[new_ind,:]
            
        return centroids
        
    def _nearest_centroids(self, X, centroids):
        """Finds the nearest centroids to the samples in X,
        returns the indices of the nearest centroids and their distance
        """
        n_samples = X.shape[0]
        n_centroids = centroids.shape[0]
        dists = np.zeros((n_samples, n_centroids))
        for i in range(n_centroids):
            delta_vec = X-np.outer(np.ones(n_samples), centroids[i])
            dists[:,i] = np.linalg.norm(delta_vec, axis=1)
        min_dists = np.min(dists, axis=1)
        indices = np.argmin(dists, axis=1)
        return indices, min_dists
    
    
    def fit(self, X, callback=None, init=None):
        n_samples = X.shape[0]
        # Initialization
        if not np.any(init):
            centroids = self.initmethod(X)
        else:
            centroids = init
        self.centroids = centroids
        
        old_assignement = np.ones(n_samples)
        new_assignment = np.zeros(n_samples)
        iterations = 0
        figs_iter = []
        
        while np.any(old_assignement-new_assignment != 0):
            if callback:
                fig = callback(X, self, f'k=3, Iteration {iterations}')
                figs_iter.append(fig)
            old_assignement = new_assignment
            new_assignment, _ = self._nearest_centroids(X, centroids)
            for i in range(self.k):
                assigned_samples = np.argwhere(new_assignment == i)
                self.centroids[i] = np.mean(X[assigned_samples, :], axis=0)
            iterations += 1
        return figs_iter
    
    def predict(self, X):
        assignment, _ = self._nearest_centroids(X, self.centroids)
        return assignment

def plot_clustering(X, clusterer, title):
    labels = clusterer.predict(X)
    centroids = clusterer.centroids

    data = pd.DataFrame(X, columns=['x_0', 'x_1'])
    col_labels = pd.Series(labels, name='Label')
    df = pd.concat([data, col_labels], axis=1)

    colors_pts = ['lightblue', 'lightgreen', 'lightsalmon', 'lightgray']
    colors_cen = ['b', 'g', 'r', 'k']
    fig = plt.figure()
    for i, label in enumerate(np.unique(labels)):
        df[labels==i].plot.scatter(x='x_0', y='x_1', c=colors_pts[i], 
                                   ax=fig.gca(), label=label, marker='o')
        plt.scatter(x=centroids[i,0], y=centroids[i,1], marker='+', c=colors_cen[i])
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.title(title)
    return fig

#%% read data
data = pd.read_csv('blobs.csv')

X = data.loc[:,['0', '1']].to_numpy()

#%% test with random initialization
km2 = KMeans(2, init='random')
km3 = KMeans(3, init='random')
km4 = KMeans(4, init='random')

km2.fit(X)
km3.fit(X)
km4.fit(X)
      
figk2 = plot_clustering(X, km2, '$k=2$')
figk3 = plot_clustering(X, km3, '$k=3$')
figk4 = plot_clustering(X, km4, '$k=4$')

#%% plot during iteration
init = X[[33,130,340],:] # these are the initial centroids
km3 = KMeans(3, init='random')
figskm3 = km3.fit(X, callback=plot_clustering, init=init)

#%% kmeans++ during iteration
kmpp3 = KMeans(3, init='k-means++')
figskpp3 = km3.fit(X, callback=plot_clustering)

#%% MNIST dataset        
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train[0:10000, :, :], [10000, 28*28])
y_train = y_train[0:10000]

mnist_clusterer = KMeans(10, init='k-means++')
mnist_clusterer.fit(x_train)
#%% plot histogram and centroids
mnist_centroids = np.reshape(mnist_clusterer.centroids, [10, 28, 28])
mnist_labels = mnist_clusterer.predict(x_train)

label_mapping = np.zeros(10, dtype=int)

# map from kmeans label to true label by finding the most frequent label 
# in the cluster
for i in range(10):
    label_mapping[i] = np.argmax(np.bincount(y_train[mnist_labels==i]))

histogram = plt.figure()
bins = np.arange(11) - 0.5
plt.hist(mnist_labels, bins=bins, alpha=0.8, label='kMeans')
plt.hist(y_train, bins=bins, alpha=0.4, label='Truth')
plt.xticks(range(10))
plt.xlabel('Cluster label')
plt.ylabel('Frequency')
plt.legend()

fig_centroids = plt.figure(figsize=(8,9))
for i in range(10):
    if i == 9:
        ax = plt.subplot(4,3,11)
    else:
        ax = plt.subplot(4,3,i+1)
    plt.imshow(mnist_centroids[i,...])
    ax.set_axis_off()
    plt.title(f'$\hat{{y}} = {label_mapping[i]}$')

#%% save figures
try:
    figk2.savefig('figs/ex4_3_k2.pdf', bbox_inches='tight')
    figk3.savefig('figs/ex4_3_k3.pdf', bbox_inches='tight')
    figk4.savefig('figs/ex4_3_k4.pdf', bbox_inches='tight')
    
    for i, f in enumerate(figskm3):
        f.savefig(f'figs/ex4_3_k_iter_{i}.pdf', bbox_inches='tight')
    for i, f in enumerate(figskpp3):
        f.savefig(f'figs/ex4_3_kpp_iter_{i}.pdf', bbox_inches='tight')
    histogram.savefig('figs/ex4_3_hist.pdf', bbox_inches='tight')
    fig_centroids.savefig('figs/ex4_3_centroids.pdf', bbox_inches='tight')
except:
    print('Error while saving figures')
