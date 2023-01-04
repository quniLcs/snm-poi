import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

# from utils.data_cvt import str2date, str2date_Bk


def prepareFoursquareYear(year):
    return year - 2012


def prepareBrightkiteYear(year):
    return (year - 2008) / 2


def prepareNodeToEmbedding(dataset, mode):
    assert dataset in ('Foursquare_TKY', 'Foursquare_NYC', 'Foursquare_NYC_LCS', 'Brightkite_x',
                       'Foursquare_TKY_no_u', 'Foursquare_TKY_no_v', 'Foursquare_TKY_no_u_no_v',
                       'Brightkite_x_no_u', 'Brightkite_x_no_v', 'Brightkite_x_no_u_no_v')
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


def prepareUserToTrajectory(dataset, time2date, prepareYear):
    assert dataset in ('Foursquare_TKY', 'Foursquare_NYC', 'Foursquare_NYC_LCS', 'Brightkite_x',
                       'Foursquare_TKY_no_u', 'Foursquare_TKY_no_v', 'Foursquare_TKY_no_u_no_v',
                       'Brightkite_x_no_u', 'Brightkite_x_no_v', 'Brightkite_x_no_u_no_v')

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
                prepareYear(date.year),
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


def prepareUserToRecommend(dataset):
    assert dataset in ('Foursquare_TKY', 'Foursquare_NYC', 'Foursquare_NYC_LCS', 'Brightkite_x',
                       'Foursquare_TKY_no_u', 'Foursquare_TKY_no_v', 'Foursquare_TKY_no_u_no_v',
                       'Brightkite_x_no_u', 'Brightkite_x_no_v', 'Brightkite_x_no_u_no_v')

    with open('../data/%s_user_re_i.pkl' % dataset, 'rb') as file:
        index2recommend = pickle.load(file)
    with open('../data/%s_user_ii.pkl'   % dataset, 'rb') as file:
        user2index = pickle.load(file)
    index2user = {index: user for user, index in user2index.items()}

    user2recommend = dict()
    for index, recommend in index2recommend.items():
        user2recommend[index2user[index]] = recommend

    with open('../data/%s_user_re.pkl' % dataset, 'wb') as file:
        pickle.dump(user2recommend, file)


class Trajectory(Dataset):
    def __init__(self, name):
        assert name in ('Foursquare_TKY', 'Foursquare_NYC', 'Foursquare_NYC_LCS', 'Brightkite_x',
                        'Foursquare_TKY_no_u', 'Foursquare_TKY_no_v', 'Foursquare_TKY_no_u_no_v',
                        'Brightkite_x_no_u', 'Brightkite_x_no_v', 'Brightkite_x_no_u_no_v')

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
    # prepareNodeToEmbedding('Foursquare_NYC_LCS', 'venue')
    # prepareNodeToEmbedding('Foursquare_NYC_LCS', 'user')
    # prepareNodeToEmbedding('Brightkite_x', 'venue')
    # prepareNodeToEmbedding('Brightkite_x', 'user')
    # prepareNodeToEmbedding('Foursquare_TKY_no_u', 'venue')
    # prepareNodeToEmbedding('Foursquare_TKY_no_u', 'user')
    # prepareNodeToEmbedding('Foursquare_TKY_no_v', 'venue')
    # prepareNodeToEmbedding('Foursquare_TKY_no_v', 'user')
    # prepareNodeToEmbedding('Foursquare_TKY_no_u_no_v', 'venue')
    # prepareNodeToEmbedding('Foursquare_TKY_no_u_no_v', 'user')
    # prepareNodeToEmbedding('Brightkite_x_no_u', 'venue')
    # prepareNodeToEmbedding('Brightkite_x_no_u', 'user')
    # prepareNodeToEmbedding('Brightkite_x_no_v', 'venue')
    # prepareNodeToEmbedding('Brightkite_x_no_v', 'user')
    # prepareNodeToEmbedding('Brightkite_x_no_u_no_v', 'venue')
    # prepareNodeToEmbedding('Brightkite_x_no_u_no_v', 'user')

    # prepareUserToTrajectory('Foursquare_TKY', time2date = str2date, prepareYear = prepareFoursquareYear)
    # prepareUserToTrajectory('Foursquare_NYC', time2date = str2date, prepareYear = prepareFoursquareYear)
    # prepareUserToTrajectory('Foursquare_NYC_LCS', time2date = str2date, prepareYear = prepareFoursquareYear)
    # prepareUserToTrajectory('Brightkite_x', time2date = str2date_Bk, prepareYear = prepareBrightkiteYear)
    # prepareUserToTrajectory('Foursquare_TKY_no_u', time2date = str2date, prepareYear = prepareFoursquareYear)
    # prepareUserToTrajectory('Foursquare_TKY_no_v', time2date = str2date, prepareYear = prepareFoursquareYear)
    # prepareUserToTrajectory('Foursquare_TKY_no_u_no_v', time2date = str2date, prepareYear = prepareFoursquareYear)
    # prepareUserToTrajectory('Brightkite_x_no_u', time2date = str2date_Bk, prepareYear = prepareBrightkiteYear)
    # prepareUserToTrajectory('Brightkite_x_no_v', time2date = str2date_Bk, prepareYear = prepareBrightkiteYear)
    # prepareUserToTrajectory('Brightkite_x_no_u_no_v', time2date = str2date_Bk, prepareYear = prepareBrightkiteYear)

    prepareUserToRecommend('Foursquare_TKY')
    prepareUserToRecommend('Foursquare_NYC')
    prepareUserToRecommend('Foursquare_NYC_LCS')
    prepareUserToRecommend('Brightkite_x')
