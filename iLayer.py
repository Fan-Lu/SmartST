import torch
import torch.nn as nn
import numpy as np


class iLayer(nn.Module):
    def __init__(self, nb_chl):
        # self.output_dim = output_dim
        super(iLayer, self).__init__()
        self.fc = nn.Linear(in_features=nb_chl, out_features=nb_chl)

    def forward(self, input):
        c = input.size(1)
        h = input.size(2)
        w = input.size(3)
        input = input.view(-1, c * h * w)
        output = self.fc(input)
        output = output.view(-1, c, h, w)
        return output
