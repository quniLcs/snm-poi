# data format:
#   Foursquare:
#       userId -> str
#       venueId -> str
#       venueCategoryId -> str
#       venueCategory -> str
#       latitude, longitude -> float
#       timezoneOffset -> int 
#       utcTimestamp -> str

import os
import random
import numpy as np
from tqdm import tqdm

class FourSquare():
    
    def __init__(self,
                 root="../data",
                 type="TKY",
                 debug=False,
                 load=False):
        '''
        Args:
            root: The root of data dir, including dir "Foursquare".
            type: "NYC" or "TKY".
        '''
        
        data_path = os.path.join(root, "Foursquare", "dataset_TSMC2014_%s.csv" % type)
        head = np.loadtxt(data_path, delimiter=',', max_rows=1, dtype=str)
        print("Loading...")
        raw_data = np.loadtxt(data_path, delimiter=',', skiprows=1, dtype=str, max_rows=1000 if debug else None)
        print("End loading raw data...")
        n_records = raw_data.shape[0]
        
        self.head = head
        self.raw_data = raw_data
        self.n_records = n_records
        
        # Get some valid information.
        user_id_list = sorted(list(set(raw_data[:, 0].tolist())), key=lambda x: int(x))   
        venue_id_list = sorted(list(set(raw_data[:, 1].tolist())))
        venue_id2idx = {venue_id: idx for idx, venue_id in enumerate(venue_id_list)}
        
        self.user_id_list = user_id_list
        self.venue_id2idx = venue_id2idx
        self.venue_id_list = venue_id_list
        
        # We get trajectories for users and POIs.
        traj_dict = {userId: [[], []] for userId in user_id_list}             # First list record id, second for timestamp.
        venue_dict = {venueId: None for venueId in venue_id_list}
        visited_dict = {venueId: [[], []] for venueId in venue_id_list}
        
        print("Get trajectories!")
        
        for rec in tqdm(raw_data):
            
            userId, venueId, timezoneOffset, utcTimestamp = rec[0], rec[1], rec[6], rec[7]
            venue_data = rec[2:6]
            
            traj_dict[userId][0].append(venueId)
            traj_dict[userId][1].append(utcTimestamp)
            visited_dict[venueId][0].append(userId)
            visited_dict[venueId][1].append(utcTimestamp)
            
            if venue_dict[venueId] is None:
                venue_dict[venueId] = venue_data
            
        self.traj_dict = traj_dict
        self.visited_dict = visited_dict
        self.venue_dict = venue_dict
        
        self.n_users = len(traj_dict)
        self.n_venues = len(venue_dict)
        
        print("Number of users: %d; Number of venues: %d" % (self.n_users, self.n_venues))
        
        
    def simulate(self,
                 n_trajs=10000,
                 length=100,
                 workers=16):
        '''
        Args:
            n_trajs: The number of simulated trajs.
            length: The length of each traj.
        Returns:
            A list, containing "n_trajs" lists.
        '''
        
        def step(node, node_category):
            
            if node_category == "user":
                next_node = random.choice(self.traj_dict[node][0])
                next_category = "venue"
            else:
                next_node = random.choice(self.visited_dict[node][0])
                next_category = "user"
            
            return next_node, next_category
        
        
        trajs = []
        
        for traj_idx in tqdm(range(n_trajs)):
            
            traj = []
            p = random.random()
            node = random.choice(self.user_id_list) if p < .5 else random.choice(self.venue_id_list)
            node_category = "user" if p < .5 else "venue"
            
            traj.append(node)
            
            for _ in range(length-1):
                
                node, node_category = step(node, node_category)
                traj.append(node)
            
            trajs.append(traj)
                
        return trajs
            
        


if __name__ == '__main__':
    
    data_root = "../data"
    dataset = FourSquare(data_root)
    trajs = dataset.simulate(n_trajs=10000, length=1000)