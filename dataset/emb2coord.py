import pickle
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


def prepareFoursquareVenueToCoordinate(city):
    """
    Since the latitudes and longitudes do not keep the same for a single venue all the time,
    store all the latitudes and longitudes and compute the mean for later usage.
    This function will be called only once.
    """
    dataset = np.loadtxt('../data/Foursquare/dataset_TSMC2014_%s.csv'  % city,
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

    with open('../data/Foursquare_%s_venue_xy.pkl' % city, 'wb') as file:
        pickle.dump(coordinate, file)

    # with open('../data/Foursquare_%s_venue_xy.pkl' % city, 'rb') as file:
    #     coordinate = pickle.load(file)

    coordinate_values = np.stack(coordinate.values())
    coordinate_mean = np.mean(coordinate_values, axis = 0)
    coordinate_std = np.std(coordinate_values, axis = 0)

    print(coordinate_mean)  # [35.67766454 139.7094122 ]
    print(coordinate_std)   # [ 0.06014593   0.07728908]

    for venueid in coordinate.keys():
        coordinate[venueid] -= coordinate_mean
        coordinate[venueid] /= coordinate_std

    with open('../data/Foursquare_%s_venue_tg.pkl' % city, 'wb') as file:
        pickle.dump(coordinate, file)


def prepareBrightkiteVenueToCoordinate():
    with open('../data/Brightkite_x_venue_wv.pkl', 'rb') as file:
        embedding = pickle.load(file)

    dataset = np.loadtxt('../data/Brightkite/Brightkite_totalCheckins.txt',
                         delimiter = '\t', skiprows = 0, dtype = str)

    coordinate = dict()
    for dataline in tqdm(dataset):
        try:
            venueid = dataline[4]
            latitude = float(dataline[2])
            longitude = float(dataline[3])
        except:
            # print(dataline)
            continue

        if venueid in embedding.keys():
            if venueid in coordinate.keys():
                assert coordinate[venueid] == (latitude, longitude)
            else:
                coordinate[venueid] = (latitude, longitude)

    for venueid in coordinate.keys():
        coordinate[venueid] = np.array(coordinate[venueid])

    with open('../data/Brightkite_x_venue_xy.pkl', 'wb') as file:
        pickle.dump(coordinate, file)

    # with open('../data/Brightkite_x_venue_xy.pkl', 'rb') as file:
    #     coordinate = pickle.load(file)

    coordinate_values = np.stack(coordinate.values())
    coordinate_mean = np.mean(coordinate_values, axis = 0)
    coordinate_std = np.std(coordinate_values, axis = 0)

    print(coordinate_mean)
    print(coordinate_std)

    for venueid in coordinate.keys():
        coordinate[venueid] -= coordinate_mean
        coordinate[venueid] /= coordinate_std

    with open('../data/Brightkite_x_venue_tg.pkl', 'wb') as file:
        pickle.dump(coordinate, file)


class EmbeddingAndCoordinate(Dataset):
    def __init__(self, name, mode):
        """
        A Pytorch Dataset Sub-class
        :param name: 'Foursquare_TKY', 'Foursquare_NYC', 'Foursquare_NYC_LCS' or 'Brightkite_x'
        :param mode: 'venue' or 'user'
        """
        assert name in ('Foursquare_TKY', 'Foursquare_NYC', 'Foursquare_NYC_LCS', 'Brightkite_x')
        assert mode in ('venue', 'user')

        with open('data/%s_%s_wv.pkl' % (name, mode), 'rb') as file:
            embedding = pickle.load(file)

        if name == 'Foursquare_NYC_LCS':
            name = 'Foursquare_NYC'

        if mode == 'venue':
            with open('data/%s_venue_tg.pkl' % name,  'rb') as file:
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


if __name__ == '__main__':
    # prepareFoursquareVenueToCoordinate('TKY')
    prepareFoursquareVenueToCoordinate('NYC')
    # prepareBrightkiteVenueToCoordinate()
