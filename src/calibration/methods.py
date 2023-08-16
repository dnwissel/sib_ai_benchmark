import torch
from torch import nn, optim
from torch.nn import functional as F


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class VectorScaling(nn.Module):
    def __init__(self, logits_len):
        super().__init__()
        self.W = torch.diag(nn.Parameter(torch.ones(logits_len) * 1.5))
        self.b = nn.Parameter(torch.zeros(logits_len) + 0.1)
        self.params = [self.W, self.b]

    def forward(self, logits):
        logits = torch.matmul(logits, self.W) + self.b
        return F.softmax(logits, dim=-1)

class MatrixScaling(nn.Module):
    def __init__(self, logits_len):
        super().__init__()
        self.layer = nn.Linear(logits_len, logits_len)
        self.params = [self.layer.state_dict()['weight'], self.layer.state_dict()['bias']]

    def forward(self, logits):
        return F.softmax(self.layer(logits), dim=-1)


class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.params = [self.temperature]

    def forward(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)

