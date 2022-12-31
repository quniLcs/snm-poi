import os
import numpy as np

from dataset import dataset
from model import *
from utils.plot import *

class trainer():
    
    def __init__(self,
                 data_root="./data",
                 dataset_type="Foursquare",
                 city="TKY"):
        '''
        Args:
            data_root: The root of data dir, including dir "Foursquare".
            dataset_type: "Foursquare" or ""
            city: if use Foursquare, "TKY" or "NYC".
        '''
        
        # Get dataset.
        if dataset_type == "Foursquare":
            self.dataset = dataset.FourSquare(data_root,
                                              city,
                                              debug=False)
        else:
            raise NotImplementedError
        
        # Train GCN.
        self._train_GNN()
        
        
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
        
        

if __name__ == '__main__':
    
    runner = trainer()
    user_wv = runner.user_wv
    plot_kmeans_tsne(user_wv, n_clusters=5)