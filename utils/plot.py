import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *
from utils.data_cvt import *

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
    
    
def plot_users_visit(user_id_list,
                     dataset,
                     save_path,
                     date_interval=None,
                     animation=None):
    '''
    Plot the places visited by selected users.
    Args:
        user_id_list: Contain the id of user to plot.
        dataset: FourSquare.
        save_path: The path to save the vis result.
        date_interval: [start_date, end_date].
        animation: "None", "day", "hour" or "month".
    '''
    
    rec_dict = {'lat': [], 'lon': [], 'day': []}
    
    for userId in user_id_list:
        
        traj, timestamps = dataset.traj_dict[userId]
        
        for venueId, timestamp in zip(traj, timestamps):
            
            date = str2date(timestamp)
            if date_interval is None or (date > date_interval[0] and date < date_interval[1]):
                venue_data = dataset.venue_dict[venueId]
                rec_dict['lat'].append(float(venue_data[-2]))
                rec_dict['lon'].append(float(venue_data[-1]))
                rec_dict['day'].append(date.toordinal())
        
    fig = px.density_mapbox(rec_dict,
                            lon='lon', lat='lat',
                            radius=3,
                            animation_frame=animation,
                            center={'lat': 35.67, 'lon': 139.71})
    fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
    fig.update_layout(mapbox_style="stamen-terrain")   
    
    # fig.show() 
    fig.write_html(save_path)
    

def stat_for_venue_category(user_id_list,
                            dataset,
                            save_path,
                            top_k=20):
    '''
    Stat for users' venue categories.
    Args:
        user_id_list: Contain the id of user to plot.
        dataset: FourSquare.
        save_path: The path to save the vis result.
        top_k: Show top_k categories.
    '''
    
    venue_category_list = dataset.venue_category_list
    venue_dict = dataset.venue_dict
    stat_dict = {venue_category: 0 for venue_category in venue_category_list}
    
    for userId in user_id_list:
        
        traj, timestamps = dataset.traj_dict[userId]
        
        for venueId, timestamp in zip(traj, timestamps):
            
            venue_category = venue_dict[venueId][1]
            stat_dict[venue_category] += 1
    
    if top_k is not None:
        stat_dict = {k : v for k, v in sorted(list(stat_dict.items()), key=lambda x: x[1], reverse=True)[:top_k]}
        venue_category_list = list(stat_dict.keys())
    
    data_dict = {"venue_category": [venue_category for venue_category in venue_category_list],\
                 "freq": [stat_dict[venue_category] for venue_category in venue_category_list]}
    fig = px.bar(data_dict, x='venue_category', y='freq')
    
    fig.write_image(save_path)