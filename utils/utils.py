import numpy as np
from datetime import datetime
import sklearn.cluster
import sklearn.manifold
from sklearn.metrics import silhouette_score

def str2date(utcTimestamp):
    
    time_data = utcTimestamp.split(' ')
    year, month, day, time = time_data[-1], time_data[1], time_data[2], time_data[3]
    hour, minute, second = time.split(':')
    
    return datetime(year, month, day, hour, minute, second)


def kmeans(wv,
           n_clusters):
    '''
    Args:
        wv: A dict. dict[userId/venueId] = vec.
        n_clusters: The number of clusters.
    Returns:
        labels result.
    '''
    
    key2idx = {key: idx for idx, key in enumerate(wv.keys())}
    idx2key = {idx: key for idx, key in enumerate(wv.keys())}
    n_recs = len(wv.keys())
    
    X = np.stack([wv[idx2key[idx]] for idx in range(n_recs)], axis=0)      # [N, k]. k is the size of vec.
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(X)
    
    labels =  kmeans.labels_
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X, labels)
    label_dict = {}                                                        # label_dict[userId] = label
    label_result = {label: [] for label in range(n_clusters)}              # label_result[label] = users
    for idx in range(n_recs):
        label_dict[idx2key[idx]] = labels[idx]
        label_result[labels[idx]].append(idx2key[idx])
        
    return label_dict, label_result, inertia, sil_score


def tsne(wv,
         n_components=2):
    '''
    Args:
        wv: A dict. dict[userId/venueId] = vec.
        n_components: The dim of the embedded space.
    Returns:
        A dict like wv. dict[userId/venueId] = embedded vec.
    '''
    
    key2idx = {key: idx for idx, key in enumerate(wv.keys())}
    idx2key = {idx: key for idx, key in enumerate(wv.keys())}
    n_recs = len(wv.keys())
    
    X = np.stack([wv[idx2key[idx]] for idx in range(n_recs)], axis=0)    # [N, k]. k is the size of vec.
    X_embedded = sklearn.manifold.TSNE(n_components=n_components).fit_transform(X) # [N, n_components].
    
    embedded_wv = {}
    for idx in range(n_recs):
        embedded_wv[idx2key[idx]] = X_embedded[idx, :]
        
    return embedded_wv