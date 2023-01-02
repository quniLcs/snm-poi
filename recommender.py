import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data_cvt import str2date


class Trajectory(Dataset):
    def __init__(self):
        # Save trajectory data from Foursquare dataset
        # with open('data/Foursquare_TKY_user_tr.pkl', 'wb') as file:
        #     pickle.dump(last_traj_dict, file)

        with open('data/Foursquare_TKY_user_tr.pkl', 'rb') as file:
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
            self.venue_embedding = pickle.load(file)
        with open('data/Foursquare_TKY_user_wv.pkl',  'rb') as file:
            self.user_embedding  = pickle.load(file)

        self.input2hidden = nn.Linear(vector_size * 2, vector_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, venue, time, hiddens):
        venue_embedding = self.venue_embedding[venue]
        time_embedding = self.time_embedding(time)

        inputs = venue_embedding + time_embedding
        combined = torch.cat((inputs, hiddens), dim = 1)
        hiddens = self.input2hidden(combined)
        output = self.softmax(output)
        return output, hiddens


def load(batch_size, num_workers, shuffle):
    dataset = Trajectory()
    print('Loaded %d trajectories' % len(dataset))

    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    print('With batch size %3d, %4d iterations per epoch\n' % (batch_size, len(dataloader)))

    return dataloader


if __name__ == '__main__':
    batch_size = 128
    num_workers = 4

    dataloader = load(batch_size, num_workers, shuffle = True)
