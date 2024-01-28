import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class VectorScaling(nn.Module):
    def __init__(self, logits_len):
        super().__init__()
        self.W = nn.Parameter(torch.ones(logits_len) * 1.0)
        self.b = nn.Parameter(torch.zeros(logits_len) + 0.1)
        self.params = [self.W, self.b]
        # self.params = [self.W]

    def forward(self, logits):
        logits = logits * self.W + self.b
        # logits = logits * self.W
        # logits = logits / self.W
        # logits = torch.matmul(logits, self.W)
        # return F.softmax(logits, dim=-1)
        # print(self.W)
        return logits


class MatrixScaling(nn.Module):
    def __init__(self, logits_len):
        super().__init__()
        self.layer = nn.Linear(logits_len, logits_len)
        self.params = [self.layer.state_dict()['weight'],
                       self.layer.state_dict()['bias']]

    def forward(self, logits):
        return self.layer(logits)


class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.5))
        self.params = [self.temperature]

    def forward(self, logits):
        # self.temperature = torch.clamp(self.temperature, min=1e-5)
        # min_temp = nn.Parameter(self.temperature.clamp(min=1e-5))
        # self.temperature = min_temp
        # self.params = [self.temperature]

        return logits / self.temperature  # logSoftmax in CrossEntropyLoss()
        # return F.softmax(/logits / self.temperature, dim=-1)
