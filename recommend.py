import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


class Recommender(nn.Module):
    def __init__(self, vector_size = 64):
        super().__init__()

        self.input2hidden = nn.Linear(vector_size * 2, vector_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hiddens):
        combined = torch.cat((inputs, hiddens), 1)
        hiddens = self.input2hidden(combined)
        output = self.i2o(combined)
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
