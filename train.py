import os
import pickle
import numpy as np

from dataset import dataset
from model import *
from utils.plot import *
from utils.utils import *

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
            raise NotImplementedError
        
        # Train GNN to get wv.
        if not load_wv:
            self._train_GNN()
            if save_wv:
                self._save_wv()
        else:
            self._load_wv()
        
        
    def _train_GNN(self,
                   n_trajs=100000,
                   length=1000):
        '''
        Train GCN using deepwalk.
        Args:
            n_trajs: The number of simulated trajs.
            length: The length of each traj.
        '''
        trajs = self.dataset.simulate(n_trajs, length)
        self.GNN = Word2Vec(trajs, vector_size=8)
        
        self.user_wv = {userId: self.GNN.wv[userId] for userId in self.dataset.user_id_list}
        self.venue_wv = {venueId: self.GNN.wv[venueId] for venueId in self.dataset.venue_id_list}
        
    
    def _save_wv(self):
        '''
        Save the wv of users and venues.
        '''
        
        assert "user_wv" in dir(self) and "venue_wv" in dir(self)
        
        user_wv_path = os.path.join(self.data_root, "%s_%s_user_wv.pkl" % (self.dataset_type, self.city))
        venue_wv_path = os.path.join(self.data_root, "%s_%s_venue_wv.pkl" % (self.dataset_type, self.city))
        
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
    
    runner = trainer(load_wv=True, save_wv=False)
    user_wv = runner.user_wv
    plot_kmeans_tsne(user_wv, n_clusters=5)
    _, label_result = kmeans(user_wv, n_clusters=5)
    import pdb; pdb.set_trace()