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


class weather(nn.Module):
    def __init__(self, input_vector_size):
        super(weather, self).__init__()
        self.input_vector = input_vector_size
        self.linear = nn.Linear(in_features=self.input_vector, out_features= 49)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=5)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=5, stride=2)
        self.deconv5 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5, stride=3)
        self.deconv6 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4)
        self.relu = nn.ReLU()

    def forward(self, weather_vector):
        weather = self.relu(self.linear(weather_vector)).view(-1, 1, 7, 7)
        weather = self.relu(self.deconv1(weather))
        weather = self.relu(self.deconv2(weather))
        weather = self.relu(self.deconv3(weather))
        weather = self.relu(self.deconv4(weather))
        weather = self.relu(self.deconv5(weather))
        weather = self.deconv6(weather)

        return weather


net = weather(8)


if __name__ == '__main__':
    a = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
    a = Variable(torch.from_numpy(a).float())
    out = net(a)
    print(out)