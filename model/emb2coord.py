from torch import nn


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
