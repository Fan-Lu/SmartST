import os
import sys
from Environment_V2 import environment
import torch
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

from Agent.ac import Actor, Critic

actor = Actor(A_DIM=8).cuda()
critic = Critic().cuda()

a_opt = optim.Adam(actor.parameters(), lr=0.001)
c_opt = optim.Adam(critic.parameters(), lr=0.001)

ENV = environment.env([21, 14], [45, 87], 999, plot=False)

action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
saved_dict = "saved_model"
saved_fig = "saved_figure"

GAMMA = 0.99
TAU = 1.0
EnCOEF = 0.01
max_times = 100

if __name__ == '__main__':
    actor.train()
    critic.train()

    value_point = ENV.data_base.value_point
    length = len(value_point)
    episode = 0
    np.random.seed(1)

    p_loss = []
    v_loss = []
    reward_record = []

    p_loss_tmp = []
    v_loss_tmp = []
    reward_record_tmp = []

    while True:
        a = np.random.randint(0, length)
        b = np.random.randint(0, length)
        Time = np.random.randint(0, 10000)
        if a==b:
            print('Zhao equals SillyB')
            continue
        s, valid_action = ENV.reset(start_loc=value_point[a], target=value_point[b], time=Time)
        if np.sum(valid_action) == 0:
            print("shibai")
            continue
        a_cx = Variable(torch.zeros(1, 256)).cuda()
        a_hx = Variable(torch.zeros(1, 256)).cuda()

        c_cx = Variable(torch.zeros(1, 256)).cuda()
        c_hx = Variable(torch.zeros(1, 256)).cuda()

        all_rewards = []
        all_values = []
        all_entropies = []
        all_lprobs = []

        for step in range(max_times):
            s = Variable(torch.from_numpy(np.array(s))).view(1, 3, 100, 100).float().cuda()
            value, (c_hx, c_cx) = critic((s, (c_hx, c_cx)))
            probs, (a_hx, a_cx) = actor((s, (a_hx, a_cx)))

            mask = Variable(torch.from_numpy(valid_action)).cuda().view(1, 8)
            masked_probs = probs * mask

            action = masked_probs.multinomial(1)
            lporbs = torch.log(probs)
            log_prob = lporbs.gather(1, action)
            entropy = -(log_prob * probs).sum(1)

            real_action = action_dic[int(action.cpu().data.numpy())]
            s_, r, done, [_, _, valid_action], success = ENV.step(real_action)   # True: Read terminal

            if np.sum(valid_action) == 0: # used to deal with the environment's dirty data.
                break

            all_rewards.append(r)
            all_values.append(value)
            all_entropies.append(entropy)
            all_lprobs.append(log_prob)

            s = s_

            # print('Episode: {} Step: {} Aciton: {}'.format(episode, step, real_action))

            # if success:
            #     if not os.path.exists('/home/exx/Lab/SmartST/model_saved_rl'):
            #         os.mkdir('/home/exx/Lab/SmartST/model_saved_rl')
            #     torch.save(actor.state_dict(), '/home/exx/Lab/SmartST/model_saved_rl/' + 'suc_model_{:d}.pth'.format(episode))

            if done or step == max_times-1:
                episode += 1
                break

        R = Variable(torch.zeros(1, 1)).cuda()
        if not done:
            s = Variable(torch.from_numpy(np.array(s))).view(1, 3, 100, 100).float().cuda()
            value, (_, _) = critic((s, (c_hx, c_cx)))
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

        p_loss_tmp.append(policy_loss.data)
        v_loss_tmp.append(value_loss.data)
        reward_record_tmp.append(r)

        if (episode+1) % 50 == 0:
            tmp_v_loss = sum(v_loss_tmp)/v_loss_tmp.__len__()
            tmp_p_loss = sum(p_loss_tmp) / p_loss_tmp.__len__()
            tmp_reward_record = sum(reward_record_tmp)/reward_record_tmp.__len__()
            print("Episode: {}, In last 50 episode, average value loss:{},"
                  "average policy loss:{},average reward:{}".format(episode, tmp_v_loss,tmp_p_loss,tmp_reward_record))

            v_loss.append(tmp_v_loss)
            p_loss.append(tmp_p_loss)
            reward_record.append(tmp_reward_record)
            v_loss_tmp = []
            p_loss_tmp = []
            reward_record_tmp = []


        if (episode + 1) % 20000 == 0:
            if not os.path.exists(saved_dict):
                os.mkdir(saved_dict)
            torch.save(actor.state_dict(), os.path.join(os.path.join(os.getcwd(), saved_dict), 'actor_model_{:d}.pth'.format(episode)))
            torch.save(critic.state_dict(), os.path.join(os.path.join(os.getcwd(), saved_dict), 'critic_model_{:d}.pth'.format(episode)))

            if not os.path.exists(saved_fig):
                os.mkdir(saved_fig)
            path = os.path.join(os.getcwd(), saved_fig)

            plt.title("Policy loss for {} episode".format(episode))
            plt.plot(p_loss)
            plt.xlabel("every 50 episode")
            plt.ylabel("every 50 episode average policy loss")
            plt.savefig(os.path.join(path, "policy_loss_{}".format(episode)))
            plt.close()
            p_loss = []

            plt.title("value loss for {} episode".format(episode))
            plt.plot(v_loss)
            plt.xlabel("every 50 episode")
            plt.ylabel("every 50 episode average value loss")
            plt.savefig(os.path.join(path, "value_loss_{}".format(episode)))
            plt.close()
            v_loss = []

            plt.title("reward for {} episode".format(episode))
            plt.plot(reward_record)
            plt.xlabel("every 50 episode")
            plt.ylabel("every 50 episode average reward")
            plt.savefig(os.path.join(path, "reward_record_{}".format(episode)))
            plt.close()
            reward_record = []