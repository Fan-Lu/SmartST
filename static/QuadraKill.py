import os
import sys
from Environment_V2 import environment_static as environment
import torch
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch.nn.functional as F
import random

from Agent.ac import Actor, Critic

device = torch.device("cuda:0")
actor = Actor(A_DIM=8).to(device)
critic = Critic().cuda().to(device)

a_opt = optim.Adam(actor.parameters(), lr=0.00001)
c_opt = optim.Adam(critic.parameters(), lr=0.00001)

ToImag = transforms.ToPILImage()
data_norm = transforms.Normalize(0, 1)

ENV = environment.env(start_loc=[21, 14], target=[35, 67], time=10, plot=False)

action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
saved_dict = "saved_model"
saved_fig = "saved_figure"

GAMMA = 0.99
TAU = 1.0
EnCOEF = 0.01
max_times = 200

dic_pair = []

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

    success = True
    # a = np.random.randint(0, length)  # start point
    # b = np.random.randint(0, length)  # end point

    failure_count = 0

    while True:
        # if success or failure_count > 100:
        start_index = np.random.randint(0, length)  # start point
        target_index = np.random.randint(0, length)  # end point
        Time = np.random.randint(0, 10000)

        failure_count = 0
        if start_index == target_index:
            print('Failed in sampling target and goal!')
            continue
        # else:
        failure_count += 1

        s, valid_action = ENV.reset(start_loc=value_point[int(start_index)], target=value_point[int(target_index)], time=10)
        if np.sum(valid_action) == 0:
            print("Failure")
            continue
        a_cx = torch.zeros(1, 256).to(device)
        a_hx = torch.zeros(1, 256).to(device)

        c_cx = torch.zeros(1, 256).to(device)
        c_hx = torch.zeros(1, 256).to(device)

        all_rewards = []
        all_values = []
        all_entropies = []
        all_lprobs = []

        for step in range(max_times):

            s = np.array(s)
            mean = np.resize([np.mean(s[i]) for i in range(3)], (3, 1, 1))
            std = np.resize([np.std(s[i]) for i in range(3)], (3, 1, 1))
            s = (s - mean) / std # Data Normalization
            s = torch.from_numpy(s).view(1, 3, 100, 100).float().to(device)

            value, (c_hx, c_cx) = critic((s, (c_hx, c_cx)))
            probs, (a_hx, a_cx) = actor((s, (a_hx, a_cx)))

            mask = torch.from_numpy(valid_action).view(1, 8).to(device)
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

            # print('Episode: {} Step: {} Target: {} Goal: {} Aciton: {}'.
            #       format(episode, step, value_point[int(start_index)],
            #              value_point[int(target_index)], real_action))

            # if success:
            #     if not os.path.exists('/home/exx/Lab/SmartST/model_saved_rl'):
            #         os.mkdir('/home/exx/Lab/SmartST/model_saved_rl')
            #     torch.save(actor.state_dict(), '/home/exx/Lab/SmartST/model_saved_rl/' + 'suc_model_{:d}.pth'.format(episode))

            if done or step == max_times-1:
                episode += 1
                break

        R = Variable(torch.zeros(1, 1)).cuda()
        if not done:
            s = Variable(torch.from_numpy(np.array(s))).view(1, 3, 100, 100).float().to(device)
            value, (_, _) = critic((s, (c_hx, c_cx)))
            R = value
        all_values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).to(device)
        for i in reversed(range(len(all_rewards))):
            R = GAMMA * R + all_rewards[i]
            advantage = R - all_values[i]
            value_loss += 0.5 * advantage.pow(2)

            a = all_values[i+1]
            #   Generalized Advantage Estimation
            delta_t = np.float(all_rewards[i]) + GAMMA * all_values[i+1] - all_values[i]

            gae = gae * GAMMA * TAU + delta_t
            policy_loss = policy_loss - all_lprobs[i] * gae - EnCOEF * all_entropies[i]

        print('Value Loss: {} Policy Loss: {}'.format(np.max(value_loss.cpu().data.numpy()), np.max(policy_loss.cpu().data.numpy())))
        actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        a_opt.step()

        critic.zero_grad()
        value_loss.backward(retain_graph=True)
        c_opt.step()

        p_loss_tmp.append(policy_loss.data)
        v_loss_tmp.append(value_loss.data)
        reward_record_tmp.append(sum(all_rewards)/all_rewards.__len__())

        if (episode+1) % 20 == 0:
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


        if (episode + 1) % 500 == 0:
            if not os.path.exists(saved_dict):
                os.mkdir(saved_dict)
            torch.save(actor.state_dict(), os.path.join(os.path.join(os.getcwd(), saved_dict), 'actor_model_{:d}.pth'.format(episode)))
            torch.save(critic.state_dict(), os.path.join(os.path.join(os.getcwd(), saved_dict), 'critic_model_{:d}.pth'.format(episode)))

            if not os.path.exists(saved_fig):
                os.mkdir(saved_fig)
            path = os.path.join(os.getcwd(), saved_fig)

            plt.title("Policy loss for {} episode".format(episode))
            plt.plot(p_loss)
            plt.xlabel("every 200 episode")
            plt.ylabel("every 200 episode average policy loss")
            plt.savefig(os.path.join(path, "policy_loss_{}".format(episode)))
            plt.close()
            p_loss = []

            plt.title("value loss for {} episode".format(episode))
            plt.plot(v_loss)
            plt.xlabel("every 200 episode")
            plt.ylabel("every 200 episode average value loss")
            plt.savefig(os.path.join(path, "value_loss_{}".format(episode)))
            plt.close()
            v_loss = []

            plt.title("reward for {} episode".format(episode))
            plt.plot(reward_record)
            plt.xlabel("every 200 episode")
            plt.ylabel("every 200 episode average reward")
            plt.savefig(os.path.join(path, "reward_record_{}".format(episode)))
            plt.close()
            reward_record = []
