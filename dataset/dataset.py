# data format:
#   Foursquare:
#       userId -> str
#       venueId -> str
#       venueCategoryId -> str
#       venueCategory -> str
#       latitude, longitude -> float
#       timezoneOffset -> int 
#       utcTimestamp -> str
#   BrightKite:
#       userId -> str
#       utcTimestamp -> str
#       latitude -> str
#       longitude -> str
#       venueId -> str

import os
import pickle
import random
import numpy as np
import multiprocessing
from tqdm import tqdm
from copy import deepcopy

class FourSquare():
    
    def __init__(self,
                 root="../data",
                 type="TKY",
                 debug=False,
                 load_geo=True,
                 load_usr=True,
                 load_xy=True):
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
        self.city = type
        self.dataset_type = "Foursquare"
        self.raw_data = raw_data
        self.n_records = n_records
        self.data_root = root
        
        # Get some valid information.
        user_id_list = sorted(list(set(raw_data[:, 0].tolist())), key=lambda x: int(x))   
        venue_id_list = sorted(list(set(raw_data[:, 1].tolist())))
        venue_category_list = sorted(list(set(raw_data[:, 3].tolist())))
        venue_id2idx = {venue_id: idx for idx, venue_id in enumerate(venue_id_list)}
        idx2venue_id = {idx: venue_id for idx, venue_id in enumerate(venue_id_list)}
        
        self.user_id_list = user_id_list
        self.venue_id2idx = venue_id2idx
        self.idx2venue_id = idx2venue_id
        self.venue_id_list = venue_id_list
        self.venue_category_list = venue_category_list
        
        # We get trajectories for users and POIs.
        traj_dict = {userId: [[], []] for userId in user_id_list}             # First list record id, second for timestamp.
        venue_dict = {venueId: None for venueId in venue_id_list}
        visited_dict = {venueId: [[], []] for venueId in venue_id_list}       # First list record id, second for timestamp.
        complete_traj_dict = {userId: [[], []] for userId in user_id_list}        # Same format as traj_dict, recording the last 3 poi of each traj.
        
        print("Get trajectories!")
        
        for rec in raw_data:
            
            userId, venueId, timezoneOffset, utcTimestamp = rec[0], rec[1], rec[6], rec[7]
            
            complete_traj_dict[userId][0].append(venueId)
            complete_traj_dict[userId][1].append(utcTimestamp)
            
        assert min([len(_[0]) for _ in complete_traj_dict.values()]) > 3
        
        last_traj_dict = {k: [v[0][-3:], v[1][-3:]] for k, v in complete_traj_dict.items()}
        
        for rec in tqdm(raw_data, leave=False, ncols=80):
            
            userId, venueId, timezoneOffset, utcTimestamp = rec[0], rec[1], rec[6], rec[7]
            venue_data = rec[2:6]
            
            if venue_dict[venueId] is None:
                venue_dict[venueId] = venue_data
                        
            last_traj = last_traj_dict[userId]
            if (venueId, utcTimestamp) in zip(last_traj[0], last_traj[1]):
                # That means test set.
                # pass
                continue
            
            traj_dict[userId][0].append(venueId)
            traj_dict[userId][1].append(utcTimestamp)
            visited_dict[venueId][0].append(userId)
            visited_dict[venueId][1].append(utcTimestamp)

        self.complete_traj_dict = complete_traj_dict
        self.last_traj_dict = last_traj_dict
        self.traj_dict = traj_dict
        self.visited_dict = visited_dict
        self.venue_dict = venue_dict
        
        self.n_users = len(user_id_list)
        self.n_venues = len(venue_id_list)
        
        print("Number of users: %d; Number of venues: %d" % (self.n_users, self.n_venues))
        
        # Load the top-k venues.
        if load_geo:
            load_path = os.path.join(root, "Foursquare_%s_topk_close_venue.pkl" % (self.city))
            with open(load_path, "rb") as f:
                self.top_k_venue_dict = pickle.load(f)
        
        if load_usr:
            load_path = os.path.join(root, "Foursquare_%s_topk_close_user.pkl" % (self.city))
            with open(load_path, "rb") as f:
                self.top_k_user_dict = pickle.load(f)   
                
        if load_xy:
            load_path = os.path.join(root, "Foursquare_%s_user_xy.csv" % (self.city))
            raw_user_xy = np.loadtxt(load_path, delimiter=',', skiprows=1, dtype=str)
            user_xy = {}
            for rec in raw_user_xy:
                user_xy[rec[0]] = rec[1:3]
            self.user_xy = user_xy
                
        test = [len(_[0]) for _ in visited_dict.values()]
        
         
    def simulate(self,
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
                p = random.random()
                if p < .5 and len(self.traj_dict[node][0]) != 0:
                    next_node = random.choice(self.traj_dict[node][0])
                    next_category = "venue"                    
                else:
                    print("Warning")
                    next_node = random.choice(self.top_k_user_dict[node])[0]
                    next_category = "user"
            else:
                p = random.random()
                if p < .5 and len(self.visited_dict[node][0]) != 0:
                    next_node = random.choice(self.visited_dict[node][0])
                    next_category = "user"
                else:
                    # print("Warning")
                    next_node = random.choice(self.top_k_venue_dict[node])
                    next_category = "venue"                    
            
            return next_node, next_category
        
        
        trajs = []
        
        for node in tqdm(self.user_id_list, ncols=80):
            
            traj = []
            node_category = "user"
            
            traj.append(node)
            
            for _ in range(length-1):
                
                node, node_category = step(node, node_category)
                traj.append(node)
            
            trajs.append(traj)
            
        for node in tqdm(self.venue_id_list, ncols=80):
            
            traj = []
            node_category = "venue"
            
            traj.append(node)
            
            for _ in range(length-1):
                
                node, node_category = step(node, node_category)
                traj.append(node)
            
            trajs.append(traj)                
                
                
        return trajs
    
    
class BrightKite():
    
    def __init__(self, 
                 root="../data",
                 debug=False,
                 load_geo=True,
                 load_xy=True):
        '''
        Args: The same as FourSquare.
        '''
        
        rec_data_path = os.path.join(root, "Brightkite", "Brightkite_filtered_Checkins.txt")
        edge_data_path = os.path.join(root, "Brightkite", "Brightkite_filtered_edges.txt")
        
        print("Loading...")
        raw_recs = np.loadtxt(rec_data_path, dtype=str, max_rows=1000 if debug else None)
        raw_edges = np.loadtxt(edge_data_path, dtype=str, max_rows=1000 if debug else None)
        print("End loading!")
        
        self.raw_recs = raw_recs
        self.raw_edges = raw_edges
        self.city = "x"
        self.dataset_type = "Brightkite"
        
        # Get some valid information.
        user_id_list = sorted(list(set(raw_recs[:, 0].tolist())), key=lambda x: int(x))   
        venue_id_list = sorted(list(set(raw_recs[:, -1].tolist())))
        venue_id2idx = {venue_id: idx for idx, venue_id in enumerate(venue_id_list)}
        idx2venue_id = {idx: venue_id for idx, venue_id in enumerate(venue_id_list)}
        
        self.user_id_list = user_id_list
        self.venue_id2idx = venue_id2idx
        self.idx2venue_id = idx2venue_id
        self.venue_id_list = venue_id_list
        
        # We get trajectories for users and POIs.
        traj_dict = {userId: [[], []] for userId in user_id_list}             # First list record id, second for timestamp.
        venue_dict = {venueId: None for venueId in venue_id_list}
        visited_dict = {venueId: [[], []] for venueId in venue_id_list}       # First list record id, second for timestamp.
        complete_traj_dict = {userId: [[], []] for userId in user_id_list}        # Same format as traj_dict, recording the last 3 poi of each traj.
        
        print("Get trajectories!")
        
        for rec in raw_recs:
            
            userId, venueId, utcTimestamp = rec[0], rec[-1], rec[1]
            
            complete_traj_dict[userId][0].append(venueId)
            complete_traj_dict[userId][1].append(utcTimestamp)
            
        # assert min([len(_[0]) for _ in last_traj_dict.values()]) > 3
        
        last_traj_dict = {k: [v[0][-3:] if len(v[0]) > 5 else [],\
                              v[1][-3:] if len(v[0]) > 5 else []] for k, v in complete_traj_dict.items()}
        
        for rec in tqdm(raw_recs, leave=False, ncols=80):
            
            userId, venueId, utcTimestamp = rec[0], rec[-1], rec[1]
            venue_data = rec[2:4]    # lat, lon.
            
            if venue_dict[venueId] is None:
                venue_dict[venueId] = venue_data
                        
            last_traj = last_traj_dict[userId]
            if (venueId, utcTimestamp) in zip(last_traj[0], last_traj[1]):
                # That means test set.
                # pass
                continue
            
            traj_dict[userId][0].append(venueId)
            traj_dict[userId][1].append(utcTimestamp)
            visited_dict[venueId][0].append(userId)
            visited_dict[venueId][1].append(utcTimestamp)

        self.complete_traj_dict = complete_traj_dict
        self.last_traj_dict = last_traj_dict
        self.traj_dict = traj_dict
        self.visited_dict = visited_dict
        self.venue_dict = venue_dict
        
        self.n_users = len(user_id_list)
        self.n_venues = len(venue_id_list)
        
        print("Number of users: %d; Number of venues: %d" % (self.n_users, self.n_venues))      
        
        # Load the top-k venues.
        if load_geo:
            load_path = os.path.join(root, "Brightkite_topk_close_venue.pkl")
            with open(load_path, "rb") as f:
                self.top_k_venue_dict = pickle.load(f)    
        
        # Process the top-k users.
        top_k_user_dict = {userId: [] for userId in self.user_id_list}
        for edge in raw_edges:
            user_from, user_to = edge
            top_k_user_dict[user_from].append(user_to)
        self.top_k_user_dict = top_k_user_dict
        
        if load_xy:
            load_path = os.path.join(root, "Brightkite_%s_user_xy.csv" % (self.city))
            raw_user_xy = np.loadtxt(load_path, delimiter=',', skiprows=1, dtype=str)        
            user_xy = {}
            for rec in raw_user_xy:
                user_xy[rec[0]] = rec[1:3]
            self.user_xy = user_xy                
                
                
    def simulate(self,
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
                p = random.random()
                if p < .5 and len(self.traj_dict[node][0]) != 0:
                    next_node = random.choice(self.traj_dict[node][0])
                    next_category = "venue"                    
                else:
                    if len(self.top_k_user_dict[node]) == 0:
                        next_node = random.choice(self.traj_dict[node][0])
                        next_category = "venue"                      
                    else:   
                        next_node = random.choice(self.top_k_user_dict[node])
                        next_category = "user"
            else:
                p = random.random()
                if p < .5 and len(self.visited_dict[node][0]) != 0:
                    next_node = random.choice(self.visited_dict[node][0])
                    next_category = "user"
                else:
                    if len(self.top_k_venue_dict[node]) == 0:
                        next_node = random.choice(self.visited_dict[node][0])
                        next_category = "user"           
                    else:
                        next_node = random.choice(self.top_k_venue_dict[node])
                        next_category = "venue"                                 
            
            return next_node, next_category
        
        
        trajs = []
        
        for node in tqdm(self.user_id_list, ncols=80):
            
            traj = []
            node_category = "user"
            
            traj.append(node)
            
            for _ in range(length-1):
                
                node, node_category = step(node, node_category)
                traj.append(node)
            
            trajs.append(traj)
            
        for node in tqdm(self.venue_id_list, ncols=80):
            
            traj = []
            node_category = "venue"
            
            traj.append(node)
            
            for _ in range(length-1):
                
                node, node_category = step(node, node_category)
                traj.append(node)
            
            trajs.append(traj)                
                
                
        return trajs                        
        

if __name__ == '__main__':
    
    data_root = "../data"
    # dataset = FourSquare(data_root, debug=False)
    # trajs = dataset.simulate(n_trajs=100, length=100)
    dataset = BrightKite(data_root)
    trajs = dataset.simulate(length=100)