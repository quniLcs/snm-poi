import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.data_cvt import str2date


def prepareNodeToEmbedding(mode):
    assert mode in ('venue', 'user')
    with open('data/Foursquare_TKY_%s_wv.pkl' % mode, 'rb') as file:
        node2embedding = pickle.load(file)

    node2index = dict()
    embeddings = list()
    for index, (node, embedding) in enumerate(node2embedding.items()):
        node2index[node] = index
        embeddings.append(embedding)
    embeddings = np.stack(embeddings)

    with open('data/Foursquare_TKY_%s_ii.pkl' % mode, 'wb') as file:
        pickle.dump(node2index, file)
    with open('data/Foursquare_TKY_%s_em.pkl' % mode, 'wb') as file:
        pickle.dump(embeddings, file)


def prepareUserToTrajectory():
    with open('data/Foursquare_TKY_user_tr.pkl',  'rb') as file:
        user2trajectory = pickle.load(file)
    with open('data/Foursquare_TKY_venue_ii.pkl', 'rb') as file:
        venue2index = pickle.load(file)
    with open('data/Foursquare_TKY_user_ii.pkl',  'rb') as file:
        user2index  = pickle.load(file)

    index2trajectory = dict()
    for user, trajectory in user2trajectory.items():
        venues = list()
        dates = list()

        for venue, time in zip(*trajectory):
            venues.append(venue2index[venue])
            date = str2date(time)
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

    with open('data/Foursquare_TKY_user_tr_i.pkl', 'wb') as file:
        pickle.dump(index2trajectory, file)


class Trajectory(Dataset):
    def __init__(self):
        # Save trajectory data from Foursquare dataset
        # with open('data/Foursquare_TKY_user_tr.pkl', 'wb') as file:
        #     pickle.dump(last_traj_dict, file)

        with open('data/Foursquare_TKY_user_tr_i.pkl', 'rb') as file:
            self.dataset = pickle.load(file)
        # self.dataset = {index: self.dataset[index] for index in range(5)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


if __name__ == '__main__':
    prepareNodeToEmbedding(mode = 'venue')
    prepareNodeToEmbedding(mode = 'user')
    prepareUserToTrajectory()
