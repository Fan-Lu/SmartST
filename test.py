import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import numpy as np

class Net(nn.Module):
    def __init__(self, nb_chl):
        # self.output_dim = output_dim
        super(Net, self).__init__()
        weight = np.random.random(nb_chl[1:])
        self.weight = Parameter(torch.from_numpy(weight).float())

    def forward(self, input):
        output = torch.mul(self.weight, input)
        return output

net = Net((None, 2, 32, 32)).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

if __name__ == '__main__':
    a = Variable(torch.randn(3, 2, 32, 32)).cuda()
    b = Variable(torch.randn(3, 2, 32, 32)).cuda()
    loss_save = []
    for i in range(1000):
        out = net(a)
        optimizer.zero_grad()
        loss = criterion(out, b)
        loss.backward()
        loss_save.append(loss.cpu().data.numpy())
        optimizer.step()
        print(i)
    plt.plot(loss_save)
    plt.show()
