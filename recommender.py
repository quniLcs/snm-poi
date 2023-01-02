import pickle

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
        index2trajectory[user2index[user]] = ([venue2index[venue] for venue in trajectory[0]], trajectory[1])

    with open('data/Foursquare_TKY_user_tr_i.pkl', 'wb') as file:
        pickle.dump(index2trajectory, file)


class Trajectory(Dataset):
    def __init__(self):
        # Save trajectory data from Foursquare dataset
        # with open('data/Foursquare_TKY_user_tr.pkl', 'wb') as file:
        #     pickle.dump(last_traj_dict, file)

        with open('data/Foursquare_TKY_user_tr_i.pkl', 'rb') as file:
            self.dataset = pickle.load(file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class TimeToEmbedding(nn.Module):
    def __init__(self, input_size = 8, hidden_size = 128, output_size = 64):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, time):
        date = str2date(time)
        inputs = [
            date.year - 2012,
            date.month / 12,
            date.day / 30,
            date.weekday() / 7,
            date.hour / 24,
            date.minute / 60,
            date.second / 60,
            date.microsecond / 1000
        ]
        return self.linear(inputs)


class Recommender(nn.Module):
    def __init__(self, vector_size = 64):
        super().__init__()
        with open('data/Foursquare_TKY_venue_wv.pkl', 'rb') as file:
            self.venue2embedding = pickle.load(file)
        with open('data/Foursquare_TKY_user_wv.pkl',  'rb') as file:
            self.user2embedding  = pickle.load(file)

        self.rnn = nn.RNN(vector_size, vector_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, venue, time, hiddens):
        venue_embedding = self.venue2embedding[venue]
        time_embedding = self.time_embedding(time)

        inputs = venue_embedding + time_embedding
        hiddens = self.RNN(inputs, hiddens)
        outputs = self.softmax(outputs)
        return outputs, hiddens


def load(batch_size, num_workers, shuffle):
    dataset = Trajectory()
    print('Loaded %d trajectories' % len(dataset))

    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    print('With batch size %3d, %4d iterations per epoch\n' % (batch_size, len(dataloader)))

    return dataloader


if __name__ == '__main__':
    batch_size = 128
    num_workers = 4

    # prepareNodeToEmbedding(mode = 'venue')
    # prepareNodeToEmbedding(mode = 'user')
    # prepareUserToTrajectory()
    dataloader = load(batch_size, num_workers, shuffle = True)
