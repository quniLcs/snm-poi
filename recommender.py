import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from emb2coord import adjustlr
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


class TimeToEmbedding(nn.Module):
    def __init__(self, input_size = 8, hidden_size = 128, output_size = 64):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs):
        return self.linear(inputs)


class Recommender(nn.Module):
    def __init__(self, vector_size = 64):
        super().__init__()

        with open('data/Foursquare_TKY_venue_em.pkl', 'rb') as file:
            venue_embeddings = pickle.load(file)
        with open('data/Foursquare_TKY_user_em.pkl',  'rb') as file:
            user_embeddings = pickle.load(file)

        self.venue_embeddings = torch.tensor(venue_embeddings)
        self.user_embeddings  = torch.tensor(user_embeddings)

        self.venue2embedding = nn.Embedding.from_pretrained(self.venue_embeddings)
        self.user2embedding  = nn.Embedding.from_pretrained(self.user_embeddings)
        self.time2embedding = TimeToEmbedding()

        self.rnn = nn.RNN(vector_size, vector_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, user, venue, time):
        user_embedding = self.user2embedding(user)
        venue_embedding = self.venue2embedding(venue)
        time_embedding = self.time2embedding(time)

        hiddens = user_embedding
        inputs = venue_embedding + time_embedding

        outputs, hiddens = self.rnn(inputs, hiddens)
        outputs = outputs[:-1] - time_embedding[1:]
        outputs = torch.inner(outputs, self.venue_embeddings)
        return self.softmax(outputs)


def build(device):
    model = Recommender()
    model.to(device)

    parameter_num = 0
    for parameter in model.parameters():
        parameter_shape = torch.tensor(parameter.shape)
        parameter_num += torch.prod(parameter_shape)
    print('Number of parameter: %d\n' % parameter_num)

    return model


def load(batch_size, num_workers, shuffle):
    dataset = Trajectory()
    print('Loaded %d trajectories' % len(dataset))

    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    print('With batch size %3d, %4d iterations per epoch\n' % (batch_size, len(dataloader)))

    return dataloader


def train(model, optimizer, criterion, dataloader,
          baselr, gamma, epoch, warmup, milestone,
          device, savedir = 'recommend'):
    model.train()

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for index in range(epoch):
        losses = []
        train_correct = 0
        train_count   = 0
        test_correct  = 0
        test_count    = 0

        for iteration, (user, (venue, time)) in tqdm(enumerate(dataloader)):
            curlr = adjustlr(optimizer, baselr, gamma, index, iteration, len(dataloader), warmup, milestone)

            user = user.to(device)
            venue = venue.to(device)
            time = time.to(device)
            outputs = model(user, venue, time)
            targets = venue[:-1]

            loss = criterion(outputs, targets)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses = np.array(losses)
        loss = np.mean(losses)

        print('Epoch: %2d\tLR: %f\tLoss: %f' % (index + 1, curlr, float(loss)))
        torch.save(model, os.path.join(savedir, 'recommender_ep%d' % (index + 1)))


if __name__ == '__main__':
    seed = 123

    batch_size = 1
    num_workers = 4

    baselr = 0.001
    gamma = 0.1
    epoch = 1000
    warmup = 1
    milestone = (900,)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = build(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = baselr)
    criterion = nn.CrossEntropyLoss()

    # prepareNodeToEmbedding(mode = 'venue')
    # prepareNodeToEmbedding(mode = 'user')
    # prepareUserToTrajectory()

    dataloader = load(batch_size, num_workers, shuffle = True)
    train(model, optimizer, criterion, dataloader, baselr, gamma, epoch, warmup, milestone, device)