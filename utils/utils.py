import numpy as np
import sklearn.cluster
import sklearn.manifold

def kmeans(wv,
           n_clusters):
    '''
    Args:
        wv: A dict. dict[userId/venueId] = vec.
        n_clusters: The number of clusters.
    Returns:
        label: A dict, recording the labels.
    '''
    
    key2idx = {key: idx for idx, key in enumerate(wv.keys())}
    idx2key = {idx: key for idx, key in enumerate(wv.keys())}
    n_recs = len(wv.keys())
    
    X = np.stack([wv[idx2key[idx]] for idx in range(n_recs)], axis=0)    # [N, k]. k is the size of vec.
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(X)
    
    labels =  kmeans.labels_
    label_dict = {}
    for idx in range(n_recs):
        label_dict[idx2key[idx]] = labels[idx]
        
    return label_dict


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