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
    
    label_dict, _ = kmeans(wv, n_clusters)
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