import pickle
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


def prepareVenueToCoordinate():
    """
    Since the latitudes and longitudes do not keep the same for a single venue all the time,
    store all the latitudes and longitudes and compute the mean for later usage.
    This function will be called only once.
    """
    dataset = np.loadtxt('data/Foursquare/dataset_TSMC2014_TKY.csv',
                         delimiter = ',', skiprows = 1, dtype = str)
    coordinate_all = defaultdict(list)
    coordinate = dict()

    for dataline in tqdm(dataset):
        venueid = dataline[1]
        latitude = float(dataline[4])
        longitude = float(dataline[5])
        coordinate_all[venueid].append((latitude, longitude))

    for venueid, coordinate_list in coordinate_all.items():
        coordinate_array = np.array(coordinate_list)
        coordinate[venueid] = np.mean(coordinate_array, axis = 0)

    with open('data/Foursquare_TKY_venue_xy.pkl', 'wb') as file:
        pickle.dump(coordinate, file)

    # with open('data/Foursquare_TKY_venue_xy.pkl', 'rb') as file:
    #     coordinate = pickle.load(file)

    coordinate_values = np.stack(coordinate.values())
    coordinate_mean = np.mean(coordinate_values, axis = 0)
    coordinate_std = np.std(coordinate_values, axis = 0)

    print(coordinate_mean)  # [35.67766454 139.7094122 ]
    print(coordinate_std)   # [ 0.06014593   0.07728908]

    for venueid in coordinate.keys():
        coordinate[venueid] -= coordinate_mean
        coordinate[venueid] /= coordinate_std

    with open('data/Foursquare_TKY_venue_tg.pkl', 'wb') as file:
        pickle.dump(coordinate, file)


class EmbeddingAndCoordinate(Dataset):
    def __init__(self, mode):
        """
        A Pytorch Dataset Sub-class
        :param mode: 'venue' or 'user'
        """
        assert mode in ('venue', 'user')
        with open('data/Foursquare_TKY_%s_wv.pkl' % mode, 'rb') as file:
            embedding = pickle.load(file)

        if mode == 'venue':
            with open('data/Foursquare_TKY_venue_tg.pkl', 'rb') as file:
                coordinate = pickle.load(file)
        else:  # mode == 'user'
            coordinate = None

        self.nodeid = list(embedding.keys())
        self.dataset = dict()
        for nodeid in self.nodeid:
            if mode == 'venue':
                self.dataset[nodeid] = (embedding[nodeid], coordinate[nodeid])
            else:  # mode == 'user'
                self.dataset[nodeid] = (embedding[nodeid], np.array([0, 0]))

    def __len__(self):
        return len(self.nodeid)

    def __getitem__(self, index):
        nodeid = self.nodeid[index]
        return nodeid, self.dataset[nodeid]
