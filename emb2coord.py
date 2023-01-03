import os
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import EmbeddingAndCoordinate
from model import EmbeddingToCoordinate


def build(device):
    model = EmbeddingToCoordinate()
    model.to(device)

    parameter_num = 0
    for parameter in model.parameters():
        parameter_num += torch.numel(parameter)
    print('Number of parameter: %d\n' % parameter_num)

    return model


def load(name, mode, batch_size, num_workers, shuffle):
    dataset = EmbeddingAndCoordinate(name, mode)
    print('Loaded %d %ss' % (len(dataset), mode))

    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    print('With batch size %3d, %4d iterations per epoch\n' % (batch_size, len(dataloader)))

    return dataloader


def adjustlr(optimizer, baselr, gamma, index, iteration, batch, warmup, milestone):
    if index < warmup:
        curlr = (iteration + index * batch) / (warmup * batch) * baselr
    else:
        curlr = baselr
        for epoch in milestone:
            if index < epoch:
                break
            curlr *= gamma

    for param_group in optimizer.param_groups:
        param_group['lr'] = curlr
    return curlr


def train(model, optimizer, criterion, trainloader, dataset,
          baselr, gamma, epoch, warmup, milestone, device, savedir = 'emb2coord'):
    assert dataset in ('Foursquare_TKY', 'Foursquare_NYC', 'Brightkite_x')
    model.train()

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for index in range(epoch):
        losses = []
        # print('Epoch: %2d' % (index + 1))

        for iteration, (nodeid, (inputs, targets)) in tqdm(enumerate(trainloader)):
            curlr = adjustlr(optimizer, baselr, gamma, index, iteration, len(trainloader), warmup, milestone)

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

        print('Epoch: %2d\tLR: %f\tLoss: %f' % (index + 1, curlr, float(loss)))
        print('Example target: (%f, %f)'   % (targets[0][0], targets[0][1]))
        print('Example output: (%f, %f)\n' % (outputs[0][0], outputs[0][1]))
        torch.save(model, os.path.join(savedir, '%s_emb2coord_ep%d' % (dataset, index + 1)))


def test(model, testloader, dataset, device):
    assert dataset in ('Foursquare_TKY', 'Foursquare_NYC', 'Brightkite_x')

    # epoch = 1000
    # savedir = 'visualize'
    # model = torch.load(os.path.join(savedir, 'emb2coord_ep%d' % epoch))
    model.eval()

    coordinate = list()
    if dataset == 'Foursquare_TKY':
        coordinate_mean = torch.tensor([35.67766454, 139.7094122])
        coordinate_std = torch.tensor([0.06014593, 0.07728908])
    elif dataset == 'Foursquare_NYC':
        coordinate_mean = torch.tensor([40.75178856, -73.97417423])
        coordinate_std = torch.tensor([0.07251354, 0.09137204])
    else:  # dataset == 'Brightkite_x'
        coordinate_mean = torch.tensor([39.80630548, -105.04022534])
        coordinate_std = torch.tensor([0.15993009, 0.1458281])

    for iteration, (nodeid, (inputs, targets)) in tqdm(enumerate(testloader)):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs).to('cpu')

        outputs *= coordinate_std
        outputs += coordinate_mean
        coordinate.append((nodeid[0], str(float(outputs[0][0])), str(float(outputs[0][1]))))

    coordinate = np.array(coordinate)
    np.savetxt('data/%s_user_xy.csv' % dataset, coordinate,
               fmt = '%s', delimiter = ',', header = 'userId,latitude,longitude')


if __name__ == '__main__':
    seed = 123

    # dataset = 'Foursquare_TKY'
    dataset = 'Foursquare_NYC'
    # dataset = 'Brightkite_x'

    batch_size = 128
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
    criterion = nn.MSELoss()

    trainloader = load(dataset, 'venue', batch_size = batch_size, num_workers = num_workers, shuffle = True)
    testloader  = load(dataset, 'user',  batch_size = 1,          num_workers = num_workers, shuffle = False)

    train(model, optimizer, criterion, trainloader, dataset, baselr, gamma, epoch, warmup, milestone, device)
    test(model, testloader, dataset, device)
