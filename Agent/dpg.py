"""
Policy Gradient
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PolicyGradient(nn.Module):

    def __init__(self, A_DIM, S_DIM, lr, reward_decay):
        """

        :param A_DIM:
        :param S_DIM:
        :param lr:  Learning Rate
        :param reward_decay:
        :param act_dic: action dictionary
        """
        super(PolicyGradient, self).__init__()
        self.gamma = reward_decay
        self.lr = lr

        self.obs, self.ep_as, self.ep_rs, self.ep_pbs = [], [], [], []    # lists used to store trainsitions

        self.conv1 = nn.Conv2d(S_DIM, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 7 * 7, 256)
        self.fc = nn.Linear(256, A_DIM)

    def forward(self, input):
        state, (hx, cx) = input
        state = Variable(torch.from_numpy(np.array(state))).view(1, 3, 100, 100).float().cuda()
        state = F.elu(self.conv1(state))
        state = F.elu(self.conv2(state))
        state = F.elu(self.conv3(state))
        state = F.elu(self.conv4(state))

        state = state.view(-1, 32 * 7 * 7)
        hx, cx = self.lstm(state, (hx, cx))
        state = hx
        probs = F.softmax(self.fc(state))

        return probs, (hx, cx)

    def store_transition(self, s, a, r, probs):
        self.obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
        self.ep_pbs.append(probs)

        return self.obs, self.ep_as, self.ep_rs, self.ep_pbs

    def discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


if __name__ == '__main__':
    RL = PolicyGradient(A_DIM=8, S_DIM=3, lr=0.01, reward_decay=0.99)
    print('No Error Found in PG')