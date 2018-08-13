import os
import sys
from Environment_V2 import environment
import torch
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import imageio

from Agent.ac import Actor, Critic
from Config import *

args = GetConfiguration()
args.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) + '/SmartST/model_saved_rl/'
args.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) + '/SmartST/result_saved_rl/'

actor = Actor(A_DIM=8).cuda()
critic = Critic(A_DIM=8).cuda()

a_opt = optim.Adam(actor.parameters(), lr=0.00001)
c_opt = optim.Adam(critic.parameters(), lr=0.00001)

ENV = environment.env(start_loc=[21, 14], target=[45, 87], time=1, args=args)

action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']

GAMMA = 0.99

if __name__ == '__main__':
    actor.train()
    critic.train()

    value_point = ENV.data_base.value_point
    episode = 0
    stop_all_flag = False
    is_test = False
    # actor.eval()
    # actor.load_state_dict(torch.load('model_saved_rl/suc_model_802.pth'))

    while True:
        a = np.random.randint(0, 1000, 1).max()
        # [48, 46]
        s = ENV.reset(start_loc=value_point[15], target=[46, 60], time=a)
        s = Variable(torch.from_numpy(np.array(s)).view(1, 3, 100, 100).float()).cuda()

        a_cx = Variable(torch.zeros(1, 256)).cuda()
        a_hx = Variable(torch.zeros(1, 256)).cuda()

        c_cx = Variable(torch.zeros(1, 256)).cuda()
        c_hx = Variable(torch.zeros(1, 256)).cuda()

        save_plots = []

        for step in range(10000):
            probs, (a_hx, a_cx) = actor((s, (a_hx, a_cx)))
            action = probs.multinomial(1)
            lporbs = torch.log(probs)
            log_prob = lporbs.gather(1, action)

            real_action = action_dic[int(action.cpu().data.numpy())]
            s_, r, done, info, success = ENV.step(real_action, episode, step, is_test)   # True: Read terminal
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

            if success:
                if not os.path.exists(args.model_dir):
                    os.mkdir(args.model_dir)
                torch.save(actor.state_dict(), args.model_dir + 'suc_model_{:d}.pth'.format(episode))

            if done:
                episode += 1
                break

        ######################## Test ###########################
        if success:
            print('<=================================================>')
            print('Start Testing')

            actor.eval()
            actor.load_state_dict(torch.load('model_saved_rl/suc_model_{}.pth'.format(episode-1)))

            is_test = True

            while True:
                a = np.random.randint(0, 1000, 1).max()
                # [48, 46] [46, 60]
                s = ENV.reset(start_loc=value_point[15], target=[46, 60], time=a)
                s = Variable(torch.from_numpy(np.array(s)).view(1, 3, 100, 100).float()).cuda()

                a_cx = Variable(torch.zeros(1, 256)).cuda()
                a_hx = Variable(torch.zeros(1, 256)).cuda()

                c_cx = Variable(torch.zeros(1, 256)).cuda()
                c_hx = Variable(torch.zeros(1, 256)).cuda()

                save_plots = []

                for step in range(10000):
                    probs, (a_hx, a_cx) = actor((s, (a_hx, a_cx)))
                    action = probs.multinomial(1)
                    lporbs = torch.log(probs)
                    log_prob = lporbs.gather(1, action)

                    real_action = action_dic[int(action.cpu().data.numpy())]
                    s_, r, done, info, success = ENV.step(real_action, episode, step, is_test)  # True: Read terminal
                    s_ = Variable(torch.from_numpy(np.array(s_))).view(1, 3, 100, 100).float().cuda()
                    s = s_

                    print('Episode: {} Step: {} Aciton: {}'.format(episode, step, real_action))

                    if success:
                        # Save to GIF
                        if not os.path.exists(args.result_dir + '../FinalResults/:'):
                            os.mkdir(args.result_dir + '../FinalResults/')

                        if not os.path.exists(args.result_dir):
                            os.mkdir(args.result_dir)

                        for i in range(step):
                            save_fn = args.result_dir + 'test_episode_{}_step_{}'.format(episode, i) + '.png'
                            save_plots.append(imageio.imread(save_fn))

                        imageio.mimsave(args.result_dir + '../FinalResults/' + 'test_episode{}.gif'.format(episode), save_plots, fps=5)
                        stop_all_flag = True
                        break
                    if done:
                        break
                if stop_all_flag: break
            if stop_all_flag:
                print('all finished')
                break

