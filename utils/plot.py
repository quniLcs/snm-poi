import os
import torch
import random
import pickle
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
                     save_dir,
                     save_name,
                     date_interval=None,
                     animation=None,
                     dataset_type="Foursquare",
                     no_plot_list=["Train Station", "Subway"],
                     mode=None):
    '''
    Plot the places visited by selected users.
    Args:
        user_id_list: Contain the id of user to plot.
        dataset: FourSquare.
        save_path: The path to save the vis result.
        date_interval: [start_date, end_date].
        animation: "None", "day", "hour" or "month".
        dataset_type: str
        no_plot_list: The no plot venue categories.
        mode: "only_weekend", "only_weekdays" or None.
    '''
    
    rec_dict = {'lat': [], 'lon': [], 'day': [], 'hour': []}
    
    for userId in user_id_list:
        
        traj, timestamps = dataset.traj_dict[userId]
        
        for venueId, timestamp in zip(traj, timestamps):
            
            date = str2date(timestamp) if dataset_type == "Foursquare" else str2date_Bk(timestamp)
            if date_interval is None or (date > date_interval[0] and date < date_interval[1]):
                venue_data = dataset.venue_dict[venueId]
                if no_plot_list is not None and venue_data[1] in no_plot_list:
                    continue
                if mode == "only_weekend" and date.weekday() in [1, 2, 3, 4, 0]:
                    continue
                elif mode == "only_weekdays" and date.weekday() in [5, 6]:
                    continue
                rec_dict['lat'].append(float(venue_data[-2]))
                rec_dict['lon'].append(float(venue_data[-1]))
                rec_dict['day'].append(date.toordinal())
                rec_dict['hour'].append(date.hour // 4)
        
    fig = px.density_mapbox(rec_dict,
                            lon='lon', lat='lat',
                            radius=3,
                            animation_frame=animation,
                            zoom=9,
                            category_orders={"hour":[0,1,2,3,4,5]} if animation=="hour" else None)
    fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
    fig.update_layout(mapbox_style="stamen-terrain")   
    
    # fig.show() 
    fig.write_html(os.path.join(save_dir, "%s.html" % save_name))
    fig.write_image(os.path.join(save_dir, "%s.png" % save_name), scale=2)
    

def stat_for_venue_category(user_id_list,
                            dataset,
                            save_dir,
                            save_name,
                            top_k=20,
                            no_plot_list=["Train Station", "Subway"],
                            not_save=False):
    '''
    Stat for users' venue categories.
    Args:
        user_id_list: Contain the id of user to plot.
        dataset: FourSquare.
        save_path: The path to save the vis result.
        top_k: Show top_k categories.
    '''
    
    venue_category_list = dataset.venue_category_list
    venue_category_list = [venue_category for venue_category in venue_category_list if venue_category not in no_plot_list]
    venue_dict = dataset.venue_dict
    stat_dict = {venue_category: 0 for venue_category in venue_category_list}
    
    count = 0
    for userId in user_id_list:
        
        traj, timestamps = dataset.traj_dict[userId]
        
        for venueId, timestamp in zip(traj, timestamps):
            
            venue_category = venue_dict[venueId][1]
            if venue_category in no_plot_list:
                continue
            stat_dict[venue_category] += 1
            count += 1
    
    if top_k is not None:
        stat_dict = {k : v for k, v in sorted(list(stat_dict.items()), key=lambda x: x[1], reverse=True)[:top_k]}
        venue_category_list = list(stat_dict.keys())
    
    data_dict = {"venue_category": [venue_category for venue_category in venue_category_list],\
                 "freq": [stat_dict[venue_category] for venue_category in venue_category_list],
                 "re_freq": [stat_dict[venue_category] / count for venue_category in venue_category_list]}
    
    if not not_save:
        fig = px.bar(data_dict, x='venue_category', y='freq')
        
        fig.write_image(os.path.join(save_dir, "%s.png" % save_name))
        
    return data_dict, stat_dict
    
    
def total_stat(plot_labels,
               label_result,
               dataset,
               save_dir,
               save_name):
    
    
    def entropy(stat_list):
        stat_ar = np.array(stat_list)
        stat_ar = stat_ar + 10
        stat_ar = stat_ar / stat_ar.sum()
        return (-stat_ar * np.log(stat_ar)).sum()
    
    
    data_dict = {"venue_category": [], "re_freq": [], "label": [], "freq": []}
    valid_venue_category_list = []
    
    for label in plot_labels:
        user_id_list = label_result[label]
        
        one_data_dict, one_stat_dict = stat_for_venue_category(user_id_list,
                            dataset,
                            save_dir,
                            save_name,
                            top_k=None,
                            no_plot_list=["Train Station", "Subway"],
                            not_save=True)
        
        data_dict["venue_category"].extend(one_data_dict["venue_category"])
        data_dict["re_freq"].extend(one_data_dict["re_freq"])
        data_dict["label"].extend(len(one_data_dict["re_freq"]) * [label])
        data_dict["freq"].extend(one_data_dict["freq"])
        
        # one_stat_dict = {k : v for k, v in sorted(list(one_stat_dict.items()), key=lambda x: x[1], reverse=True)[:5]}
        # venue_category_list = list(one_stat_dict.keys())        
        # valid_venue_category_list.extend(venue_category_list)
    
    # Choose the categories to plot.
    entropy_dict = {key: [0] * len(plot_labels) for key in set(data_dict["venue_category"])}
    for idx in range(len(data_dict["venue_category"])):
        entropy_dict[data_dict["venue_category"][idx]][plot_labels.index(data_dict["label"][idx])] += data_dict["freq"][idx]
    
    # import pdb; pdb.set_trace()
    category_entropy_list = [(category, entropy(entropy_dict[category])) for category in entropy_dict.keys()]
    topk_category_entropy_list = sorted(category_entropy_list, reverse=True, key=lambda x: x[1])[:15]
    valid_venue_category_list = [category for category, _ in topk_category_entropy_list]
    # import pdb; pdb.set_trace()
    
    valid_venue_category_list = list(set(valid_venue_category_list))
    new_data_dict = {"venue_category": [], "re_freq": [], "label": []}
    for idx in range(len(data_dict["venue_category"])):
        if data_dict["venue_category"][idx] in valid_venue_category_list:
            new_data_dict["venue_category"].append(data_dict["venue_category"][idx])
            new_data_dict["re_freq"].append(data_dict["re_freq"][idx])
            new_data_dict["label"].append(data_dict["label"][idx])
    
    fig = px.bar(new_data_dict, x='venue_category', y='re_freq', color='label', barmode='group')
    fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
    fig.update_layout(mapbox_style="stamen-terrain")     
    fig.write_image(os.path.join(save_dir, "%s.png" % save_name))
    
    
def plot_csv_data(trainer,
                  label_dict,
                  save_dir,
                  save_name,
                  n_clusters):
    '''
    Args:
        trainer.
        n_clusters: The num of clusters.
    '''
    
    # First we cluster them based on embeddings.
    dataset = trainer.dataset
    user_wv = trainer.user_wv
    user_xy = dataset.user_xy
    
    # label_dict, label_result, _, __ = kmeans(user_wv, n_clusters)
    
    # Plot the users.
    rec_dict = {'id': [], 'lat': [], 'lon': [], 'label': []}
    user_id_list = dataset.user_id_list
    
    for userId in user_id_list:
        lat, lon = user_xy[userId]
        rec_dict['lat'].append(float(lat))
        rec_dict['lon'].append(float(lon))
        rec_dict['label'].append(label_dict[userId])
        rec_dict['id'].append(userId)
        
    fig = px.scatter_mapbox(rec_dict,
                            lon='lon', lat='lat',
                            zoom=9,
                            color='label',
                            mapbox_style="light")
    fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
    fig.update_layout(mapbox_style="stamen-terrain")   
    
    # fig.show() 
    fig.write_html(os.path.join(save_dir, "%s.html" % save_name))
    fig.write_image(os.path.join(save_dir, "%s.png" % save_name), scale=2)    
    
    
def plot_re_data(trainer,
                 data_root,
                 plot_num,
                 save_dir,
                 save_name,
                 top_k=3):
    
    dataset_type = trainer.dataset.dataset_type
    city = trainer.dataset.city
    
    re_data_path = os.path.join(data_root, '%s_%s_user_re.pkl' % (dataset_type, city))
    with open(re_data_path, "rb") as f:
        re_dict = pickle.load(f)
    with open(os.path.join(data_root, '%s_%s_venue_ii.pkl' % (dataset_type, city)), 'rb') as file:
        venue2index = pickle.load(file)        
    
    # import pdb; pdb.set_trace()
    index2venue = {v: k for k, v in venue2index.items()}
    re_dict = {k: [_ for _ in torch.reshape(v[:, :top_k], shape=(-1,))] for k, v in re_dict.items()}
    user_xy = trainer.dataset.user_xy
    venue_dict = trainer.dataset.venue_dict
    traj_dict = trainer.dataset.traj_dict
    last_traj_dict = trainer.dataset.last_traj_dict
    # import pdb; pdb.set_trace()
    
    user_id_list = random.choices(list(re_dict.keys()), k=plot_num)
    # user_id_list = ["2100"]
    plot_dict = {"user_id": [], "lat": [], "lon": [], "recommend": []}
    
    for userId in user_id_list:
        # # First get the origin lonlat.
        # lat, lon = user_xy[userId]
        # plot_dict['lat'].append(float(lat))
        # plot_dict['lon'].append(float(lon))
        # plot_dict["user_id"].append(userId)
        # plot_dict["recommend"].append(0)
        
        # We get the trajactoires.
        traj = last_traj_dict[userId][0]
        for venue_id in traj:
            venue_data = venue_dict[venue_id]
            plot_dict['lat'].append(float(venue_data[-2]))
            plot_dict['lon'].append(float(venue_data[-1]))            
            plot_dict["user_id"].append(userId)
            plot_dict["recommend"].append(0)            
        
        # Then get the recommend lonlat.
        for idx in range(3*top_k):
            venue_id = index2venue[re_dict[userId][idx].numpy().tolist()]
            venue_data = venue_dict[venue_id]
            plot_dict['lat'].append(float(venue_data[-2]))
            plot_dict['lon'].append(float(venue_data[-1]))            
            plot_dict["user_id"].append(userId)
            plot_dict["recommend"].append(1)
    
    # import pdb; pdb.set_trace()        
    
    fig = px.density_mapbox(plot_dict,
               lon='lon', lat='lat',
               zoom=9,
               radius=5,
               animation_frame="recommend")
    fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
    fig.update_layout(mapbox_style="stamen-terrain")       
    # fig.write_image(os.path.join(save_dir, "%s.png" % save_name))
    fig.write_html(os.path.join(save_dir, "%s.html" % save_name))