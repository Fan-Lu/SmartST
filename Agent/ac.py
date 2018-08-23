import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, S_DIM=3, A_DIM=4):
        super(Actor, self).__init__()

        self.a_conv1 = nn.Conv2d(S_DIM, 32, 3, stride=2, padding=1)
        self.a_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.a_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.a_conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.a_lstm = nn.LSTMCell(32 * 7 * 7, 256)
        self.actor_linear = nn.Linear(256, A_DIM)

    def forward(self, input):
        state, (hx, cx) = input
        state = F.elu(self.a_conv1(state))
        state = F.elu(self.a_conv2(state))
        state = F.elu(self.a_conv3(state))
        state = F.elu(self.a_conv4(state))

        state = state.view(-1, 32*7*7)
        hx, cx = self.a_lstm(state, (hx, cx))

        probs = F.softmax(self.actor_linear(hx))

        return probs, (hx, cx)


class Critic(nn.Module):
    def __init__(self, S_DIM=3, A_DIM=4):
        super(Critic, self).__init__()

        self.c_conv1 = nn.Conv2d(S_DIM, 32, 3, stride=2, padding=1)
        self.c_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.c_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.c_conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.c_lstm = nn.LSTMCell(32 * 7 * 7, 256)

        self.critic_linear = nn.Linear(256, 1)

    def forward(self, input):
        state, (hx, cx) = input
        state = F.elu(self.c_conv1(state))
        state = F.elu(self.c_conv2(state))
        state = F.elu(self.c_conv3(state))
        state = F.elu(self.c_conv4(state))

        state = state.view(-1, 32*7*7)
        hx, cx = self.c_lstm(state, (hx, cx))
        state = hx

        value = self.critic_linear(state)

        return value, (hx, cx)


if __name__ == '__main__':
    cx = Variable(torch.zeros(1, 256))
    hx = Variable(torch.zeros(1, 256))
    state = Variable(torch.randn(3, 100, 100)).view(1, 3, 100, 100)

    print('test')
