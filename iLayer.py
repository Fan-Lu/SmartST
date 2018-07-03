import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter


class iLayer(nn.Module):
    def __init__(self, nb_chl):
        # self.output_dim = output_dim
        super(iLayer, self).__init__()
        weight = np.random.random(nb_chl)
        self.weight = Parameter(torch.from_numpy(weight).float())

    def forward(self, input):
        output = torch.mul(self.weight, input)
        return output
