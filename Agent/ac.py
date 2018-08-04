import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, A_Dim=4, S_Dim=3):
        """
        :param A_Dim: Number of actions
        :param S_Dim: Dimension of input state
        :param Optimizer: Actor and Critic Net optimizer
        """
        super(ActorCritic, self).__init__()
        self.A_Dim = A_Dim
        self.S_Dim = S_Dim

        self.c_conv1 = nn.Conv2d(S_Dim, 32, 3, stride=2, padding=1)
        self.c_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.c_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.c_conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.a_lstm = nn.LSTMCell(32 * 7 * 7, 256)
        self.c_lstm = nn.LSTMCell(32 * 7 * 7, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, A_Dim)

    def Actor(self, input):
        state, (hx, cx) = input
        state = F.elu(self.a_conv1(state))
        state = F.elu(self.a_conv2(state))
        state = F.elu(self.a_conv3(state))
        state = F.elu(self.a_conv4(state))

        state = state.view(-1, 32*7*7)
        hx, cx = self.a_lstm(state, (hx, cx))
        state = hx

        action = self.actor_linear(state)

        return action, (hx, cx)

    def Critic(self, input):
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

    def Train(self, v_curr, v_next, lprobs, r, GAMMA):
        """
        one step update actor critic
        :param v_curr: current critic value
        :param v_next: next critic value
        :param lprobs: log probability of action chosen
        :param r: current rewards
        :param GAMMA: discount factor
        :return:
        """
        #   Critic Learn
        self.c_opt.zero_grad()
        td_error = r + GAMMA * v_next - v_curr
        td_error = torch.sqrt(td_error)
        td_error.backward(retain_graph=True)
        self.c_opt.step()
        # TODO: Check RL4B if optimizer of actor and critic are the same

        #   Actor Lear
        self.a_opt.zero_grad()
        exp_v = -lprobs * td_error
        exp_v.backward(retain_graph=True)
        self.a_opt.step()

    def Test(self):
        pass


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
        state = hx
        probs = F.softmax(self.actor_linear(state))

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

ac = ActorCritic(4, 3)

if __name__ == '__main__':
    cx = Variable(torch.zeros(1, 256))
    hx = Variable(torch.zeros(1, 256))
    state = Variable(torch.randn(3, 100, 100)).view(1, 3, 100, 100)
    input = (state, (hx, cx))
    value, action, (hx, cx) = ac(input)
    r = 10 # from environment

    print('test')
