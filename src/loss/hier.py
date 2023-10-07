import numpy as np
from torch import nn, optim
from torch.nn import functional
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class MCLoss(nn.Module):
    """
    input should be logits.
    """
    def __init__(self, encoder):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.encoder = encoder

    def forward(self, output, target):
        # print(target)
        target = self.encoder.transform(target.cpu().numpy())
        target = target.astype(np.double)
        target = torch.from_numpy(target).to(device)

        R = self.encoder.get_R()
        constr_output = get_constr_out(output, R)
        train_output = target*output.double()
        train_output = get_constr_out(train_output, R)
        train_output = (1-target)*constr_output.double() + target*train_output

        #MCLoss
        # print(train_output[:,self.encoder.idx_to_eval ], target[:,self.encoder.idx_to_eval])
        # mask = train_output < 0
        # train_output[mask] = 0
        loss = self.criterion(train_output[:,self.encoder.idx_to_eval ], target[:,self.encoder.idx_to_eval])
        return loss

def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    
    # x = x.to(device)
    # R = R.to(device)

    # Not enough mem in GPU, calculate on CPU
    x = x.to('cpu')
    R = R.to('cpu')

    x = torch.sigmoid(x)
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    # put back on GPU
    final_out = final_out.to(device)

    return final_out