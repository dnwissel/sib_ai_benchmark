import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class MaskBCE(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder
        # self.R = R
        self.criterion = F.binary_cross_entropy_with_logits
        # self.bid = 0

    def forward(self, output, target):
        # constr_output = get_constr_out(output, self.R)
        # train_output = target*output.double()
        # train_output = get_constr_out(train_output, self.R)
        # train_output = (1-target)*constr_output.double() + target*train_output
        train_output = output

        # Mask Loss
        loss_mask = self.encoder.get_lossMask()
        loss_mask = loss_mask.to(device)
        lm_batch = loss_mask[target]
        target = self.encoder.transform(target.cpu().numpy())
        target = target.astype(np.float32)
        target = torch.from_numpy(target).to(device)

        # #Mask target
        # lm_batch = loss_mask[target]
        # target = self.encoder.transform(target.numpy())
        # target = target.astype(np.float32)
        # target = np.where(lm_batch, target, 1)
        # target = torch.from_numpy(target).to(device)

        loss = self.criterion(train_output[:, self.encoder.idx_to_eval],
                              target[:, self.encoder.idx_to_eval], reduction='none')
        loss = lm_batch[:, self.encoder.idx_to_eval] * loss
        return loss.sum()

    def set_encoder(self, encoder):
        self.encoder = encoder


class MCLoss(nn.Module):
    """
    input should be logits.
    """

    def __init__(self, encoder=None):
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

        # MCLoss
        # print(train_output[:,self.encoder.idx_to_eval ], target[:,self.encoder.idx_to_eval])
        # mask = train_output < 0
        # train_output[mask] = 0
        loss = self.criterion(
            train_output[:, self.encoder.idx_to_eval], target[:, self.encoder.idx_to_eval])
        return loss

    def set_encoder(self, encoder):
        self.encoder = encoder


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
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim=2)
    # put back on GPU
    final_out = final_out.to(device)

    return final_out
