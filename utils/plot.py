import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *

def plot_kmeans_tsne(wv,
                     n_clusters):
    '''
    Plot users' vecs using kmeans and tsne.
    Args:
        wv: A dict. dict[userId/venueId] = vec.
        n_clusters: The number of clusters.
    '''
    
    label_dict, _, _, _ = kmeans(wv, n_clusters)
    embedded_dict = tsne(wv)
    
    X, Y, labels = [], [], []
    for key in label_dict.keys():
        
        x, y = embedded_dict[key]
        label = label_dict[key]
        
        X.append(x); Y.append(y); labels.append(label)
        
    plt.scatter(X, Y, c=labels, s=5)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
def plot_kmeans_inertia(wv,
                        k_range=[5, 10]):
    '''
    Plot the inertia of different n_clusters.
    Args:
        wv: A dict. dict[userId/venueId] = vec.
    '''
    
    inertias = []
    for k in range(k_range[0], k_range[1]+1):
        _, _, inertia, _ = kmeans(wv, k)
        inertias.append(inertia)
    
    plt.plot(range(k_range[0], k_range[1]+1), inertias, 'o-')
    plt.show()
    

def plot_kmeans_silhouette_score(wv,
                                 k_range=[5, 10]):
    '''
    Plot the silhouette score of different n_clusters.
    Args:
        wv: A dict. dict[userId/venueId] = vec.
    '''    
    
    silhouette_scores = []
    for k in range(k_range[0], k_range[1]+1):
        _, _, _, sil_score = kmeans(wv, k)
        silhouette_scores.append(sil_score)
        
    plt.plot(range(k_range[0], k_range[1]+1), silhouette_scores, 'o-')
    plt.show()    