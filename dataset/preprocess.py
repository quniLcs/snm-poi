import gc
import os
import pickle
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from dataset import FourSquare

def compute_topk_venue_dict(data_root,
                            type="TKY",
                            top_k=10):
    
    dataset = FourSquare(data_root, type)
    
    venue_dict = dataset.venue_dict
    venue_id_list = dataset.venue_id_list
    venue_id2idx, idx2venue_id = dataset.venue_id2idx, dataset.idx2venue_id
    n_venue = len(venue_id_list)
    
    # Get the pos array.
    pos_array = np.zeros((n_venue, 2), dtype=np.float32)   # [N, 2]
    for venue_id in venue_id_list:
        pos_array[venue_id2idx[venue_id], :] = [float(venue_dict[venue_id][2]), float(venue_dict[venue_id][3])]
        
    # Compute the dist.
    tops = []
    batch = 100
    part_num = n_venue // batch
    total_idx_list = list(range(n_venue))
    for part_idx in tqdm(range(part_num), ncols=80):
        idx_list = total_idx_list[part_idx*batch : (part_idx+1)*batch]
        if part_idx == (part_num - 1): idx_list = total_idx_list[part_idx*batch : ]
        X = pos_array[idx_list, :]      # [k, 2]
        dist_mat = X[:, None, :] - pos_array      # [k, N, 2]
        one_dist = np.linalg.norm(dist_mat, axis=2).astype(np.float32)   # [k, N]
        # import pdb; pdb.set_trace()
        top = np.argpartition(one_dist, top_k+1, axis=1)[:, 1:top_k+1]  # [k, top_k]
        tops.append(top)
        del one_dist
        del dist_mat
        gc.collect()
    
    tops = np.concatenate(tops, axis=0)       # [N, top_k]
    
    topk_close_venue_dict = {}
    for idx in range(n_venue):
        top = tops[idx]
        top_rec = [idx2venue_id[_] for _ in top]
        topk_close_venue_dict[idx2venue_id[idx]] = top_rec
        
    save_path = os.path.join(data_root, "Foursquare_%s_topk_close_venue.pkl" % (dataset.city))
    with open(save_path, "wb") as f:
        pickle.dump(topk_close_venue_dict, f)     
        

def compute_topk_user_dict(data_root,
                           type="TKY",
                           top_k=20):  
    
    def similarity(traj_1_idx, traj_2_idx):
        
        def dot(traj_1, traj_2):
            return ((traj_1 - traj_2) == 0).sum()


        diff = traj_1_idx.size - traj_2_idx.size
        
        if diff < 0:
            short, long = traj_1_idx, traj_2_idx
        else:
            short, long = traj_2_idx, traj_1_idx
            
        length = short.size
        
        max = 0
        for i in range(abs(diff)+1):
            score = dot(short, long[i : i+length])
            max = score if max < score else max
    
        return max
    
    
    dataset = FourSquare(data_root, type)
    
    user_id_list = dataset.user_id_list
    venue_id2idx, idx2venue_id = dataset.venue_id2idx, dataset.idx2venue_id
    traj_dict = deepcopy(dataset.traj_dict)
    
    traj_dict = {key: np.array([venue_id2idx[_] for _ in traj_dict[key][0]]) for key in user_id_list}
    
    topk_close_user_dict = {}
    for userId_i in tqdm(user_id_list, ncols=80):
        temp = []
        for userId_j in user_id_list:
            traj_i, traj_j = traj_dict[userId_i], traj_dict[userId_j]
            temp.append((userId_j, similarity(traj_i, traj_j)))
        topk_close_user_dict[userId_i] = sorted(temp, key=lambda x: x[1], reverse=True)[1:top_k+1]
    
    save_path = os.path.join(data_root, "Foursquare_%s_topk_close_user.pkl" % (dataset.city))
    with open(save_path, "wb") as f:
        pickle.dump(topk_close_user_dict, f)     
    
    

if __name__ == '__main__':
    
    data_root = "../data"
    # compute_topk_venue_dict(data_root)
    compute_topk_user_dict(data_root)