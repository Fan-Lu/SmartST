import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.optim as optim


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, in_dim, layers, num_classes=512):
        self.inplanes = 8
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.fc = nn.Linear(3136, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Actor(nn.Module):
    def __init__(self, S_DIM=3, A_DIM=4):
        super(Actor, self).__init__()

        self.a_resnet = ResNet(BasicBlock, S_DIM, [3, 3, 3, 3])
        self.a_lstm = nn.LSTMCell(512, 256)
        self.actor_linear = nn.Linear(256, A_DIM)

    def forward(self, input):
        state, (hx, cx) = input
        state = self.a_resnet(state)
        hx, cx = self.a_lstm(state, (hx, cx))

        probs = F.softmax(self.actor_linear(hx))

        return probs, (hx, cx)


class Critic(nn.Module):
    def __init__(self, S_DIM=3):
        super(Critic, self).__init__()

        self.c_resnet = ResNet(BasicBlock, S_DIM, [3, 3, 3, 3])
        self.c_lstm = nn.LSTMCell(512, 256)
        self.critic_linear = nn.Linear(256, 1)

    def forward(self, input):
        state, (hx, cx) = input
        state = self.c_resnet(state)
        hx, cx = self.c_lstm(state, (hx, cx))
        state = hx

        value = self.critic_linear(state)

        return value, (hx, cx)


if __name__ == '__main__':
    cx = Variable(torch.zeros(1, 256))
    hx = Variable(torch.zeros(1, 256))
    state = Variable(torch.randn(3, 100, 100)).view(1, 3, 100, 100)

    print('test')
