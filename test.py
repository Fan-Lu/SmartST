import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc = nn.Linear(5, 1)

    def forward(self, input):
        output = nn.Linear(5, 1).cuda()(input)
        return output

net = Net()
a = Variable(torch.randn(4, 5))
print(net(a))
