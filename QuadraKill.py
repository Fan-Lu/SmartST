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

actor = Actor(A_DIM=8).cuda()
critic = Critic(A_DIM=8).cuda()

a_opt = optim.Adam(actor.parameters(), lr=0.00001)
c_opt = optim.Adam(critic.parameters(), lr=0.00001)

ENV = environment.env([21, 14], [45, 87], 999)

action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']

GAMMA = 0.99
TAU = 1.0
EnCOEF = 0.01

if __name__ == '__main__':
    actor.train()
    critic.train()

    value_point = ENV.data_base.value_point
    episode = 0

    while True:
        s = ENV.reset(start_loc=value_point[15], target=[48, 46], time=1)

        a_cx = Variable(torch.zeros(1, 256)).cuda()
        a_hx = Variable(torch.zeros(1, 256)).cuda()

        c_cx = Variable(torch.zeros(1, 256)).cuda()
        c_hx = Variable(torch.zeros(1, 256)).cuda()

        all_rewards = []
        all_values = []
        all_entropies = []
        all_lprobs = []

        for step in range(10000):
            s = Variable(torch.from_numpy(np.array(s))).view(1, 3, 100, 100).float().cuda()
            value, (c_hx, c_cx) = critic((s, (c_hx, c_cx)))
            probs, (a_hx, a_cx) = actor((s, (a_hx, a_cx)))
            action = probs.multinomial(1)
            lporbs = torch.log(probs)
            log_prob = lporbs.gather(1, action)
            entropy = -(log_prob * probs).sum(1)

            real_action = action_dic[int(action.cpu().data.numpy())]
            s_, r, done, info, success = ENV.step(real_action)   # True: Read terminal

            all_rewards.append(r)
            all_values.append(value)
            all_entropies.append(entropy)
            all_lprobs.append(log_prob)

            s = s_

            print('Episode: {} Step: {} Aciton: {}'.format(episode, step, real_action))

            if success:
                if not os.path.exists('/home/exx/Lab/SmartST/model_saved_rl'):
                    os.mkdir('/home/exx/Lab/SmartST/model_saved_rl')
                torch.save(actor.state_dict(), '/home/exx/Lab/SmartST/model_saved_rl/' + 'suc_model_{:d}.pth'.format(episode))

            if done:
                episode += 1
                break

        R = Variable(torch.zeros(1, 1)).cuda()
        if not done:
            value, (_, _) = critic(s, (c_hx, c_cx))
            R = value
        all_values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = Variable(torch.zeros(1, 1)).cuda()
        for i in reversed(range(len(all_rewards))):
            R = GAMMA * R + all_rewards[i]
            advantage = R - all_values[i]
            value_loss += 0.5 * advantage.pow(2)

            a = all_values[i+1]
            #   Generalized Advantage Estimation
            delta_t = np.float(all_rewards[i]) + GAMMA * all_values[i+1] - all_values[i]

            gae = gae * GAMMA * TAU + delta_t
            policy_loss = policy_loss - all_lprobs[i] * gae - EnCOEF * all_entropies[i]

        actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        a_opt.step()

        critic.zero_grad()
        value_loss.backward(retain_graph=True)
        c_opt.step()

        if (episode + 1) % 1000 == 0:
            if not os.path.exists('/home/exx/Lab/SmartST/model_saved_rl'):
                os.mkdir('/home/exx/Lab/SmartST/model_saved_rl')
            torch.save(actor.state_dict(), '/home/exx/Lab/SmartST/model_saved_rl/' + 'model_{:d}.pth'.format(episode))


