import os
import sys
import numpy as np
from Environment_V2.environment_static_2 import environment
import torch
import torch
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

from Agent.ac import Actor, Critic

tra_num = 5

maps = []
game = {}
num_of_map = 5
num_of_point = 5

i = 0
for ele in [2, 10, 18, 21, 33, 34, 36, 53, 56]:
    maps.append("map_{}.npy".format(ele))
    game[maps[i]] = [[105, 180], [200, 300], [80, 351], [5, 150], [50, 205]]
    i += 1

# [80, 351], [10, 95], [1, 66]

# maps = ["map_{}.npy".format(i) for i in range(61, 3)]
#
# game = {"map_61.npy": 5,
#         "map_54.npy": [[105, 180], [200, 300], [80, 351]]}

num_channel = 3
ENV = environment(maps=maps, map_size=(50, 50), games=game, only_when_success=True, digital=num_channel!=3,
                  reward_type="two", way_back=False, running_reward=True, running_reward_interval=100)

device = torch.device("cuda:0")
actor = Actor(A_DIM=8, num_channel=num_channel)
critic = Critic(num_channel=num_channel)
success_counter = 0

def normalize(s):
    """
    Coarse normalize
    :param s: ndarray
    :return: ndarray
    """
    for i in range(len(s)):
        s[i] /= np.max(s[i])
    return s

a_opt = optim.Adam(actor.parameters(), lr=0.0005)
c_opt = optim.Adam(critic.parameters(), lr=0.0005)


action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
saved_dict = "saved_model"
saved_fig = "saved_figure"

GAMMA = 0.99
TAU = 1.0
EnCOEF = 0.01
max_times = 300

clip = 1

if __name__ == '__main__':

    file_reward = 'reward.txt'
    file_success = 'success.txt'

    if os.path.exists(file_reward):
        os.remove(file_reward)

    if os.path.exists(file_success):
        os.remove(file_success)

    actor.train()
    critic.train()

    episode = 1
    np.random.seed(1)

    p_loss = []
    v_loss = []
    reward_record = []

    p_loss_tmp = []
    v_loss_tmp = []
    reward_record_tmp = []
    plot_flag = False

    while True:
        # if episode % 10 == 0:
        #     plot_flag = True
        # else:
        #     plot_flag = False

        s, location, mask = ENV.reset(plot=plot_flag)
        s = normalize(s)

        all_rewards = []
        all_values = []
        all_entropies = []
        all_lprobs = []


        for step in range(max_times):
            s = torch.from_numpy(np.array(s)).view(1, num_channel, 50, 50).float()
            value = critic(s)
            probs = actor(s)

            mask = torch.from_numpy(mask).view(1, 8)
            masked_probs = probs * mask

            action = masked_probs.multinomial(1)  # 这里你没用masked_probs,所以会选出异常的方向
            lporbs = torch.log(probs)
            log_prob = lporbs.gather(1, action)
            entropy = -(log_prob * probs).sum(1)
            real_action = action_dic[int(action.cpu().data.numpy())]
            if step == max_times-1:
                s_, _, mask, r, success, f_r, running_mean_reward = ENV.step(real_action, last_step=True)
            else:
                s_, _, mask, r, success, f_r, running_mean_reward = ENV.step(real_action)

            all_rewards.append(r)
            all_values.append(value)
            all_entropies.append(entropy)
            all_lprobs.append(log_prob)

            s = s_
            s = normalize(s)

            print(r)

            if success:
                r = f_r * 100
                all_rewards[-1] = r
                success_counter += 1
                break

        # record rewards
        if episode % 10 == 0:
            with open("reward.txt", 'a') as f_reward:
                f_reward.write('total reward: {0}.'.format(all_rewards[-1]) + '\n')

        print("Running {0} times, and successed {1} times!".format(episode, success_counter))

        # record running time and success time
        with open("success.txt", 'a') as f_success:
            f_success.write('total time: {0}, success time: {1}'.format(episode, success_counter) + '\n')

        episode += 1

        R = torch.zeros(1, 1)
        if not success:
            s = torch.from_numpy(np.array(s)).view(1, num_channel, 50, 50).float()
            value = critic(s)
            R = value
        all_values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(all_rewards))):
            R = GAMMA * R + all_rewards[i]
            advantage = R - all_values[i]
            value_loss += 0.5 * advantage.pow(2)

            delta_t = np.float(all_rewards[i]) + GAMMA * all_values[i+1] - all_values[i]

            gae = gae * GAMMA * TAU + delta_t
            policy_loss = policy_loss - all_lprobs[i] * gae - EnCOEF * all_entropies[i]

        actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        a_opt.step()
        torch.nn.utils.clip_grad_norm(actor.parameters(), clip)

        critic.zero_grad()
        value_loss.backward(retain_graph=True)
        c_opt.step()
        torch.nn.utils.clip_grad_norm(critic.parameters(), clip)

        if (episode + 1) % 500 == 0:
            torch.save(actor.state_dict(), 'actor_model_{:d}.pth'.format(episode))
            torch.save(critic.state_dict(), 'critic_model_{:d}.pth'.format(episode))
