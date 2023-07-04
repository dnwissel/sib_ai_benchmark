import torch
from torch import nn, optim
from torch.nn import functional as F


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# TODO: add vs

class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)


