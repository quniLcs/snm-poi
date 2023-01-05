import os
import pickle
import random
import numpy as np
from datetime import datetime

from dataset import dataset
from model import *
from utils.plot import *
from utils.utils import *
from utils.stat import *

class trainer():
    
    def __init__(self,
                 data_root="./data",
                 dataset_type="Foursquare",
                 city="TKY",
                 load_wv=False,
                 save_wv=True):
        '''
        Args:
            data_root: The root of data dir, including dir "Foursquare".
            dataset_type: "Foursquare" or ""
            city: if use Foursquare, "TKY" or "NYC".
        '''
        
        # Get dataset.
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.city = city
        
        if dataset_type == "Foursquare":
            self.dataset = dataset.FourSquare(data_root,
                                              city,
                                              debug=False)
        else:
            self.dataset = dataset.BrightKite(data_root)
            self.city = 'x'
        
        # Train GNN to get wv.
        if not load_wv:
            self._train_GNN()
            if save_wv:
                self._save_wv()
        else:
            self._load_wv()
        
        
    def _train_GNN(self,
                   length=1500):
        '''
        Train GCN using deepwalk.
        Args:
            n_trajs: The number of simulated trajs.
            length: The length of each traj.
        '''
        trajs = self.dataset.simulate(length)
        self.GNN = Word2Vec(trajs, vector_size=64, workers=16, min_count=0, sg=1)
        
        self.user_wv = {userId: self.GNN.wv[userId] for userId in self.dataset.user_id_list}
        self.venue_wv = {venueId: self.GNN.wv[venueId] for venueId in self.dataset.venue_id_list}
        
    
    def _save_wv(self):
        '''
        Save the wv of users and venues.
        '''
        
        assert "user_wv" in dir(self) and "venue_wv" in dir(self)
        
        user_wv_path = os.path.join(self.data_root, "%s_%s_user_wv_no_u.pkl" % (self.dataset_type, self.city))
        venue_wv_path = os.path.join(self.data_root, "%s_%s_venue_wv_no_u.pkl" % (self.dataset_type, self.city))
        
        with open(user_wv_path, "wb") as f:
            pickle.dump(self.user_wv, f)
        
        with open(venue_wv_path, "wb") as f:
            pickle.dump(self.venue_wv, f)
            
    
    def _load_wv(self):
        '''
        Load the wv of users and venues.
        '''
        
        user_wv_path = os.path.join(self.data_root, "%s_%s_user_wv.pkl" % (self.dataset_type, self.city))
        venue_wv_path = os.path.join(self.data_root, "%s_%s_venue_wv.pkl" % (self.dataset_type, self.city))      
        
        assert os.path.exists(user_wv_path) and os.path.exists(venue_wv_path)
        
        with open(user_wv_path, "rb") as f:
            self.user_wv = pickle.load(f)
        
        with open(venue_wv_path, "rb") as f:
            self.venue_wv = pickle.load(f)
        
        

if __name__ == '__main__':
    
    runner = trainer(load_wv=False, save_wv=True, dataset_type="Brightkite")
    plot_id = '1139'
    # user_id_list = runner.dataset.user_id_list
    # user_id_list = runner.dataset.top_k_user_dict[plot_id]
    user_id_list = random.choices(runner.dataset.user_id_list, k=50)
    # user_id_list = [plot_id]
    # stat_LCS(plot_id, runner.dataset)
    # import pdb; pdb.set_trace()
    # plot_users_visit(user_id_list,
    #                  runner.dataset,
    #                  save_dir="./visualize/vis_traj/",
    #                  save_name="vis_traj_random",
    #                  date_interval=None,
    #                  dataset_type="Brightkite",
    #                  no_plot_list=None,
    #                  animation="hour")    
    # import pdb; pdb.set_trace()
    
    # user_wv = runner.user_wv
    # plot_kmeans_tsne(user_wv, n_clusters=6)
    # plot_kmeans_inertia(user_wv, k_range=[2, 10])
    # plot_kmeans_silhouette_score(user_wv, k_range=[2, 10])
    # import pdb; pdb.set_trace()
    
    # n_clusters = 5
    # label_dict, label_result, _, _ = kmeans(user_wv, n_clusters)
    # import pdb; pdb.set_trace()
    # label_dict, label_result = spectral_clustering(user_wv, 4)
    
    # date_interval = [datetime(2012, 4, 15, 0, 0, 0), datetime(2012, 7, 15, 0, 0, 0)]
    # date_interval = None
    # for label in range(n_clusters):
    #     user_id_list = label_result[label]
    #     plot_users_visit(user_id_list,
    #                      runner.dataset,
    #                      save_dir="./visualize/poi_vis/",
    #                      save_name="vis_%d_weekend" % label,
    #                      date_interval=date_interval,
    #                      animation="hour",
    #                      mode="only_weekend")
    #     plot_users_visit(user_id_list,
    #                      runner.dataset,
    #                      save_dir="./visualize/poi_vis/",
    #                      save_name="vis_%d_weekdays" % label,
    #                      date_interval=date_interval,
    #                      animation="hour",
    #                      mode="only_weekdays")        
        # import pdb; pdb.set_trace()
        # stat_for_venue_category(user_id_list,
        #                         runner.dataset,
        #                         save_dir="./visualize/stat_bar/",
        #                         save_name="bar_%d" % label,
        #                         top_k=20)
    
    
    # plot_csv_data(runner,
    #               label_dict=label_dict,
    #               save_dir="./visualize/poi_vis/",
    #               save_name="vis",
    #               n_clusters=n_clusters)
    
    # total_stat(plot_labels=[0, 1, 2, 3, 4, 5],
    #            label_result=label_result,
    #            dataset=runner.dataset,
    #            save_dir="./visualize/stat_bar/",
    #            save_name="total")
    
    # plot_re_data(runner,
    #              data_root="./data",
    #              plot_num=50,
    #              save_dir="visualize/recommend",
    #              save_name="recommend",
    #              top_k=1)