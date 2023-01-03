import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.data_cvt import str2date, str2date_Bk


def prepareNodeToEmbedding(dataset, mode):
    assert mode in ('venue', 'user')
    with open('../data/%s_%s_wv.pkl' % (dataset, mode), 'rb') as file:
        node2embedding = pickle.load(file)

    node2index = dict()
    embeddings = list()
    for index, (node, embedding) in enumerate(node2embedding.items()):
        node2index[node] = index
        embeddings.append(embedding)
    embeddings = np.stack(embeddings)

    with open('../data/%s_%s_ii.pkl' % (dataset, mode), 'wb') as file:
        pickle.dump(node2index, file)
    with open('../data/%s_%s_em.pkl' % (dataset, mode), 'wb') as file:
        pickle.dump(embeddings, file)


def prepareUserToTrajectory(dataset, time2date):
    with open('../data/%s_user_tr.pkl'  % dataset, 'rb') as file:
        user2trajectory = pickle.load(file)
    with open('../data/%s_venue_ii.pkl' % dataset, 'rb') as file:
        venue2index = pickle.load(file)
    with open('../data/%s_user_ii.pkl'  % dataset, 'rb') as file:
        user2index  = pickle.load(file)

    index2trajectory = dict()
    for user, trajectory in user2trajectory.items():
        venues = list()
        dates = list()

        for venue, time in zip(*trajectory):
            venues.append(venue2index[venue])
            date = time2date(time)
            dates.append([
                date.year - 2012,
                date.month / 12,
                date.day / 30,
                date.weekday() / 7,
                date.hour / 24,
                date.minute / 60,
                date.second / 60,
                date.microsecond / 1000
            ])

        index2trajectory[user2index[user]] = (
            torch.tensor(venues),
            torch.tensor(dates)
        )

    with open('../data/%s_user_tr_i.pkl' % dataset, 'wb') as file:
        pickle.dump(index2trajectory, file)


class Trajectory(Dataset):
    def __init__(self, name):
        assert name in ('Foursquare_TKY', 'Foursquare_NYC', 'Brightkite_x')

        # Save trajectory data from Foursquare and Brightkite dataset
        # with open('../data/Foursquare_TKY_user_tr.pkl', 'wb') as file:
        #     pickle.dump(last_traj_dict, file)
        # with open('../data/Foursquare_NYC_user_tr.pkl', 'wb') as file:
        #     pickle.dump(last_traj_dict, file)
        # with open('../data/Brightkite_x_user_tr.pkl',   'wb') as file:
        #     pickle.dump(last_traj_dict, file)

        with open('data/%s_user_tr_i.pkl' % name, 'rb') as file:
            self.dataset = pickle.load(file)
        # self.dataset = {index: self.dataset[index] for index in range(5)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


if __name__ == '__main__':
    # prepareNodeToEmbedding('Foursquare_TKY', 'venue')
    # prepareNodeToEmbedding('Foursquare_TKY', 'user')
    # prepareNodeToEmbedding('Foursquare_NYC', 'venue')
    # prepareNodeToEmbedding('Foursquare_NYC', 'user')
    # prepareNodeToEmbedding('Brightkite_x', 'venue')
    # prepareNodeToEmbedding('Brightkite_x', 'user')
    prepareUserToTrajectory('Foursquare_TKY', time2date = str2date)
    prepareUserToTrajectory('Foursquare_NYC', time2date = str2date)
    prepareUserToTrajectory('Brightkite_x', time2date = str2date_Bk)
