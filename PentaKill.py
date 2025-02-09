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

if __name__ == '__main__':
    actor.train()
    critic.train()

    value_point = ENV.data_base.value_point  # list [[x1,y1],[x2,y2]]
    episode = 0

    while True:
        s = ENV.reset(start_loc=value_point[15], target=[48, 46], time=1)
        s = Variable(torch.from_numpy(np.array(s)).view(1, 3, 100, 100).float()).cuda()

        a_cx = Variable(torch.zeros(1, 256)).cuda()
        a_hx = Variable(torch.zeros(1, 256)).cuda()

        c_cx = Variable(torch.zeros(1, 256)).cuda()
        c_hx = Variable(torch.zeros(1, 256)).cuda()

        for step in range(10000):

            ###
            # actor.mask_move = ENV.get_moveable_list()
            ###

            probs, (a_hx, a_cx) = actor((s, (a_hx, a_cx)))
            action = probs.multinomial(1)
            lporbs = torch.log(probs)
            log_prob = lporbs.gather(1, action)

            real_action = action_dic[int(action.cpu().data.numpy())]
            s_, r, done, info, success = ENV.step(real_action)   # True: Read terminal
            s_ = Variable(torch.from_numpy(np.array(s_))).view(1, 3, 100, 100).float().cuda()

            v_curr, (c_hx, c_cx) = critic((s, (c_hx, c_cx)))
            v_next, (c_hx, c_cx) = critic((s_, (c_hx, c_cx)))

            #   Critic Learn
            c_opt.zero_grad()
            td_error = np.float(r) + GAMMA * v_next - v_curr
            # td_error = torch.sqrt(td_error)
            td_error.backward(retain_graph=True)
            c_opt.step()

            #   Actor Lear
            a_opt.zero_grad()
            exp_v = -log_prob * td_error
            exp_v.backward(retain_graph=True)
            a_opt.step()

            s = s_

            print('Episode: {} Step: {} Aciton: {}'.format(episode, step, real_action))

            if done:
                episode += 1
                break

            if success:
                if not os.path.exists('/home/exx/Lab/SmartST/model_saved_rl'):
                    os.mkdir('/home/exx/Lab/SmartST/model_saved_rl')
                torch.save(actor.state_dict(), '/home/exx/Lab/SmartST/model_saved_rl/' + 'suc_model_{:d}.pth'.format(episode))

        if (episode + 1) % 1000 == 0:
            if not os.path.exists('/home/exx/Lab/SmartST/model_saved_rl'):
                os.mkdir('/home/exx/Lab/SmartST/model_saved_rl')
            torch.save(actor.state_dict(), '/home/exx/Lab/SmartST/model_saved_rl/' + 'model_{:d}.pth'.format(episode))


