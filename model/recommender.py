import pickle

import torch
import torch.nn as nn


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
    def __init__(self, vector_size = 64, criterion = 'MSELoss', device = 'cpu'):
        super().__init__()
        assert criterion in ('CrossEntropyLoss', 'MSELoss')

        with open('data/Foursquare_TKY_venue_em.pkl', 'rb') as file:
            venue_embeddings = pickle.load(file)
        with open('data/Foursquare_TKY_user_em.pkl',  'rb') as file:
            user_embeddings = pickle.load(file)

        self.venue_embeddings = torch.tensor(venue_embeddings).to(device)
        self.user_embeddings  = torch.tensor(user_embeddings).to(device)

        self.venue2embedding = nn.Embedding.from_pretrained(self.venue_embeddings)
        self.user2embedding  = nn.Embedding.from_pretrained(self.user_embeddings)
        self.time2embedding = TimeToEmbedding()

        self.rnn = nn.RNN(vector_size, vector_size, batch_first = True)
        self.softmax = nn.Softmax(dim = 2)

        self.criterion = criterion
        if self.criterion == 'CrossEntropyLoss':
            self.loss = nn.CrossEntropyLoss()
        else:  # self.criterion == 'MSELoss'
            self.loss = nn.MSELoss()

    def forward(self, user, venue, time):
        user_embedding = self.user2embedding(user)
        venue_embedding = self.venue2embedding(venue)
        time_embedding = self.time2embedding(time)

        hiddens = torch.unsqueeze(user_embedding, dim = 1)
        inputs = venue_embedding + time_embedding

        outputs, hiddens = self.rnn(inputs[:, :-3, :], hiddens)
        outputs = outputs[:, :-1, :] - time_embedding[:, 1:-3, :]
        targets = venue[:, 1:-3]

        if self.criterion == 'CrossEntropyLoss':
            outputs = torch.inner(outputs, self.venue_embeddings)
            outputs = self.softmax(outputs)
            outputs = torch.transpose(outputs, dim0 = 1, dim1 = 2)
        else:  # self.criterion == 'MSELoss'
            targets = self.venue2embedding(targets)
        loss = self.loss(outputs, targets)

        with torch.no_grad():
            outputs, _ = self.rnn(inputs[:, -3:, :], hiddens)
            outputs = torch.cat((hiddens, outputs), dim = 1)
            outputs = outputs[:, :-1, :] - time_embedding[:, -3:, :]
            targets = venue[:, -3:]

            outputs = torch.inner(outputs, self.venue_embeddings)
            _, outputs = torch.topk(outputs, k = 20, dim = 2)
            correct = torch.eq(outputs, targets.unsqueeze(dim = 2))

            correct01 = torch.sum(correct[:, :, :1])
            correct05 = torch.sum(correct[:, :, :5])
            correct10 = torch.sum(correct[:, :, :10])
            correct20 = torch.sum(correct[:, :, :20])
            count = torch.numel(targets)

        return loss, correct01, correct05, correct10, correct20, count
