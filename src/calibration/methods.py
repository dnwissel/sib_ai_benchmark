import torch
from torch import nn, optim
from torch.nn import functional as F


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# TODO: add vs

class VectorScaling(nn.Module):
    def __init__(self, logits_len):
        super().__init__()
        # self.W = nn.Parameter(torch.diag(torch.ones(1) * 1.5))
        # self.b = nn.Parameter(torch.ones(1) * 1.5)
        self.layer = nn.Linear(logits_len, logits_len)

    def forward(self, logits):
        # logits = torch.tensordot(self.W, logits, dims=1) + self.b
        return F.softmax(self.layer(logits), dim=-1)



class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)

