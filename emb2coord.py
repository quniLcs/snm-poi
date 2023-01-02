import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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


class ResBlock(nn.Module):
    def __init__(self, vector_size = 8, hidden_size = 32):
        super().__init__()
        self.relu = nn.ReLU()
        self.resblock = nn.Sequential(
            nn.Linear(vector_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vector_size),
            nn.BatchNorm1d(vector_size),
        )

    def forward(self, inputs):
        return self.relu(inputs + self.resblock(inputs))


class EmbeddingToCoordinate(nn.Module):
    def __init__(self, vector_size = 64, hidden_size = 512, output_size = 2):
        super().__init__()
        self.resblock1 = ResBlock(vector_size, hidden_size)
        self.resblock2 = ResBlock(vector_size, hidden_size)
        self.resblock3 = ResBlock(vector_size, hidden_size)
        self.resblock4 = ResBlock(vector_size, hidden_size)  # 269186
        self.linear = nn.Linear(vector_size, output_size)

    def forward(self, inputs):
        outputs = self.resblock1(inputs)
        outputs = self.resblock2(outputs)
        outputs = self.resblock3(outputs)
        outputs = self.resblock4(outputs)
        return self.linear(outputs)


def load(mode, batch_size, num_workers, shuffle):
    dataset = EmbeddingAndCoordinate(mode = mode)
    print('Loaded %d %ss' % (len(dataset), mode))

    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    print('With batch size %3d, %4d iterations per epoch\n' % (batch_size, len(dataloader)))

    return dataloader


def build(device):
    model = EmbeddingToCoordinate()
    model.to(device)

    parameter_num = 0
    for parameter in model.parameters():
        parameter_shape = torch.tensor(parameter.shape)
        parameter_num += torch.prod(parameter_shape)
    print('Number of parameter: %d\n' % parameter_num)

    return model


def adjustlr(baselr, gamma, index, iteration, batch, warmup, milestone):
    if index < warmup:
        curlr = iteration / (batch * warmup) * baselr
    else:
        curlr = baselr
        for epoch in milestone:
            if index < epoch:
                break
            curlr *= gamma

    for param_group in optimizer.param_groups:
        param_group['lr'] = curlr
    return curlr


def train(model, optimizer, criterion, trainloader,
          baselr, gamma, epoch, warmup, milestone,
          device, savedir = 'visualize'):
    model.train()

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for index in range(epoch):
        losses = []
        # print('Epoch: %2d' % (index + 1))

        for iteration, (nodeid, (inputs, targets)) in tqdm(enumerate(trainloader)):
            curlr = adjustlr(baselr, gamma, index, iteration, len(trainloader), warmup, milestone)

            inputs = inputs.to(device)
            targets = targets.to(device).to(torch.float)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            losses.append(loss.item())
            # if (iteration + 1) % 200 == 0:
            #     print('Iteration: %3d\tLR: %f\tLoss: %f' % (iteration + 1, curlr, float(loss)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses = np.array(losses)
        loss = np.mean(losses)

        print('Epoch: %2d\t\tLR: %f\tLoss: %f' % (index + 1, curlr, float(loss)))
        print('Example target: (%f, %f)'   % (targets[0][0], targets[0][1]))
        print('Example output: (%f, %f)\n' % (outputs[0][0], outputs[0][1]))
        torch.save(model, os.path.join(savedir, 'emb2coord_ep%d' % (index + 1)))


def test(model, testloader, device):
    epoch = 1000
    savedir = 'visualize'
    model = torch.load(os.path.join(savedir, 'emb2coord_ep%d' % epoch))
    model.eval()

    coordinate = list()
    coordinate_mean = torch.tensor([35.67766454, 139.7094122])
    coordinate_std = torch.tensor([0.06014593, 0.07728908])

    for iteration, (nodeid, (inputs, targets)) in tqdm(enumerate(testloader)):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs).to('cpu')

        outputs *= coordinate_std
        outputs += coordinate_mean
        coordinate.append((nodeid[0], str(float(outputs[0][0])), str(float(outputs[0][1]))))

    coordinate = np.array(coordinate)
    np.savetxt('data/Foursquare_TKY_user_xy.csv', coordinate,
               fmt = '%s', delimiter = ',', header = 'userId,latitude,longitude')


if __name__ == '__main__':
    seed = 123

    batch_size = 128
    num_workers = 4

    baselr = 0.001
    gamma = 0.1
    warmup = 1
    milestone = (900,)
    epoch = 1000

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = build(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = baselr)
    criterion = nn.MSELoss()

    # prepareVenueToCoordinate()
    trainloader = load('venue', batch_size = batch_size, num_workers = num_workers, shuffle = True)
    testloader  = load('user',  batch_size = 1,          num_workers = num_workers, shuffle = False)

    # train(model, optimizer, criterion, trainloader, baselr, gamma, epoch, warmup, milestone, device)
    test(model, testloader, device)
