import os
import sys
from Environment_V2 import environment
import torch
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from Agent.ac import Actor, Critic
from Agent.dpg import PolicyGradient

import matplotlib.pyplot as plt


ac_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
PG = PolicyGradient(A_DIM=8, S_DIM=3, lr=0.01, reward_decay=0.95).cuda()
Optimizer = optim.Adam(PG.parameters(), lr=0.001)
ENV = environment.env([2, 51], [48, 46], 1)


if __name__ == '__main__':
    value_point = ENV.data_base.value_point
    episode = 0
    PG.train()

    while True:
        s = ENV.reset(start_loc=value_point[22], target=[48, 46], time=1)

        cx = Variable(torch.zeros(1, 256)).cuda()
        hx = Variable(torch.zeros(1, 256)).cuda()

        step = 0

        while True:
            probs, (hx, cx) = PG((s, (hx, cx)))
            action = probs.multinomial(1).cpu().data.numpy()
            s_, r, done, info, success = ENV.step(ac_dic[action.max()])  # True: Read terminal
            ep_obs, ep_as, ep_rs, ep_pbs = PG.store_transition(s, action.max(), r, probs.cpu().data.numpy())
            step += 1

            print('Episode: {} Step: {} Aciton: {}'.format(episode, step, ac_dic[action.max()]))

            if done:
                episode += 1
                ep_rs_sum = sum(PG.ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                PG.zero_grad()
                # discount and normalize episode reward
                discounted_ep_rs_norm = PG.discount_and_norm_rewards()
                discounted_ep_rs_norm = Variable(torch.from_numpy(discounted_ep_rs_norm).float())

                # to maximize total reward (log_p * R) is to minimize -(log_p * R)
                all_act_prob = Variable(torch.from_numpy(np.array(ep_pbs)), requires_grad=True).view(-1, 8)

                all_acts = torch.from_numpy(np.array(ep_as))
                all_acts = all_acts.view(-1, 1)
                one_hot = torch.zeros(all_acts.size(0), 8).scatter_(1, all_acts, 1)
                one_hot = Variable(torch.transpose(one_hot, 0, 1))

                neg_log_prob = -torch.trace(torch.log(all_act_prob) @ one_hot)
                loss = torch.sum(neg_log_prob * discounted_ep_rs_norm).cuda()
                loss.backward()
                Optimizer.step()

                ep_as, ep_pbs, ep_obs, ep_rs = [], [], [], []

                # if episode % 100 == 0:
                #     plt.close()
                #     plt.plot(discounted_ep_rs_norm.data.numpy())    # plot the episode vt
                #     plt.xlabel('episode steps')
                #     plt.ylabel('normalized state-action value')
                #     plt.show()

                break

            s = s_
