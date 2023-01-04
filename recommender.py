import os
# import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from emb2coord import adjustlr
from dataset import Trajectory
from model import Recommender


def build(dataset, device):
    model = Recommender(dataset = dataset, device = device)
    model.to(device)

    for module in model.modules():
        if type(module) == nn.Linear:
            nn.init.orthogonal_(module.weight)

    parameter_num = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter_num += torch.numel(parameter)
    print('Number of parameter: %d\n' % parameter_num)

    return model


def load(name, batch_size, num_workers, shuffle):
    dataset = Trajectory(name)
    print('Loaded %d trajectories' % len(dataset))

    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle)
    print('With batch size %3d, %4d iterations per epoch\n' % (batch_size, len(dataloader)))

    return dataloader


def train(model, optimizer, dataloader, dataset,
          baselr, gamma, epoch, warmup, milestone,
          device, savedir = 'recommend'):
    # epoch = 100
    # model = torch.load(os.path.join(savedir, '%s_recommender_ep%d' % (dataset, epoch)))

    model.train()

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for index in range(epoch):
        losses = []
        # outputs = {}
        corrects01 = 0
        corrects05 = 0
        corrects10 = 0
        corrects20 = 0
        counts = 0

        for iteration, (user, (venue, time)) in tqdm(enumerate(dataloader)):
            curlr = adjustlr(optimizer, baselr, gamma, index, iteration, len(dataloader), warmup, milestone)

            user = user.to(device)
            venue = venue.to(device)
            time = time.to(device)

            loss, output, correct01, correct05, correct10, correct20, count = model(user, venue, time)
            # if output is not None:
            #     outputs[int(user)] = output[0].to(cpu)
            corrects01 += correct01
            corrects05 += correct05
            corrects10 += correct10
            corrects20 += correct20
            counts += count
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses = np.array(losses)
        loss = np.mean(losses)

        print('Epoch: %2d\tLR: %f\tLoss: %f\tAcc@1: %f\tAcc@5: %f\tAcc@10: %f\tAcc@20: %f' %
              (index + 1, curlr, float(loss),
               corrects01 / counts, corrects05 / counts,
               corrects10 / counts, corrects20 / counts))
        torch.save(model, os.path.join(savedir, '%s_recommender_ep%d' % (dataset, index + 1)))
        # with open('data/%s_user_re_i.pkl' % dataset, 'wb') as file:
        #     pickle.dump(outputs, file)
        # break


if __name__ == '__main__':
    seed = 123

    dataset = 'Foursquare_TKY'
    # dataset = 'Foursquare_NYC'
    # dataset = 'Brightkite_x'

    batch_size = 1
    num_workers = 4

    baselr = 0.001
    gamma = 0.5
    epoch = 1000
    warmup = 1
    milestone = (20, 40, 60, 80, 100)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = build(dataset, device)
    optimizer = torch.optim.Adam(model.parameters(), lr = baselr)

    dataloader = load(dataset, batch_size, num_workers, shuffle = True)
    train(model, optimizer, dataloader, dataset, baselr, gamma, epoch, warmup, milestone, device)
