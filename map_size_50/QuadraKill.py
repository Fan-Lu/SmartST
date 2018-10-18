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

maps = ["map_54.npy", "map_61.npy", "map_62.npy"]

ENV = environment(maps=maps, map_size=(50, 50), games=3, only_when_success=True, digital=True,
                  reward_type="two", way_back=True, running_reward=True, running_reward_interval=100)

device = torch.device("cuda:0")
actor = Actor(A_DIM=8)
critic = Critic()
success_counter = 0

def normalize(s):
    s[0] = s[0]/20.0
    s[1] = s[1]/50.0
    return s

a_opt = optim.Adam(actor.parameters(), lr=0.0005)
c_opt = optim.Adam(critic.parameters(), lr=0.0005)


action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
saved_dict = "saved_model"
saved_fig = "saved_figure"

GAMMA = 0.99
TAU = 1.0
EnCOEF = 0.01
max_times = 200

clip = 1

if __name__ == '__main__':
    actor.train()
    critic.train()

    episode = 0
    np.random.seed(1)

    p_loss = []
    v_loss = []
    reward_record = []

    p_loss_tmp = []
    v_loss_tmp = []
    reward_record_tmp = []

    while True:
        s, _, mask = ENV.reset(plot=True)
        s = normalize(s)


        all_rewards = []
        all_values = []
        all_entropies = []
        all_lprobs = []

        for step in range(max_times):
            try:
                s = torch.from_numpy(np.array(s)).view(1, 1, 50, 50).float()
            except:
                print("stop here s")
            value = critic(s)
            probs = actor(s)

            mask = torch.from_numpy(mask).view(1, 8)
            masked_probs = probs * mask

            try:
                action = probs.multinomial(1)
            except:
                print("stop here a")
            lporbs = torch.log(probs)
            log_prob = lporbs.gather(1, action)
            entropy = -(log_prob * probs).sum(1)
            try:
                real_action = action_dic[int(action.cpu().data.numpy())]
            except:
                print("stop here ra")
            s_, _, mask, r, success, f_r, running_mean_reward = ENV.step(real_action)

            all_rewards.append(r)
            all_values.append(value)
            all_entropies.append(entropy)
            all_lprobs.append(log_prob)

            s = s_
            s = normalize(s)

            if success:
                r = f_r
                success_counter += 1

        R = torch.zeros(1, 1)
        if not success:
            try:
                s = torch.from_numpy(np.array(s)).view(1, 1, 50, 50).float()
            except:
                print("stop here s_")
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

        # p_loss_tmp.append(policy_loss.data)
        # v_loss_tmp.append(value_loss.data)
        # reward_record_tmp.append(sum(all_rewards)/all_rewards.__len__())

        # if (episode+1) % 200 == 0:
        #     tmp_v_loss = sum(v_loss_tmp)/v_loss_tmp.__len__()
        #     tmp_p_loss = sum(p_loss_tmp) / p_loss_tmp.__len__()
        #     tmp_reward_record = sum(reward_record_tmp)/reward_record_tmp.__len__()
        #     print("Episode: {}, In last 50 episode, average value loss:{},"
        #           "average policy loss:{},average reward:{}".format(episode, tmp_v_loss,tmp_p_loss,tmp_reward_record))
        #
        #     v_loss.append(tmp_v_loss)
        #     p_loss.append(tmp_p_loss)
        #     reward_record.append(tmp_reward_record)
        #     v_loss_tmp = []
        #     p_loss_tmp = []
        #     reward_record_tmp = []

        #
        # if (episode + 1) % 50000 == 0:
        #     if not os.path.exists(saved_dict):
        #         os.mkdir(saved_dict)
        #     torch.save(actor.state_dict(), os.path.join(os.path.join(os.getcwd(), saved_dict), 'actor_model_{:d}.pth'.format(episode)))
        #     torch.save(critic.state_dict(), os.path.join(os.path.join(os.getcwd(), saved_dict), 'critic_model_{:d}.pth'.format(episode)))
        #
        #     if not os.path.exists(saved_fig):
        #         os.mkdir(saved_fig)
        #     path = os.path.join(os.getcwd(), saved_fig)
        #
        #     plt.title("Policy loss for {} episode".format(episode))
        #     plt.plot(p_loss)
        #     plt.xlabel("every 200 episode")
        #     plt.ylabel("every 200 episode average policy loss")
        #     plt.savefig(os.path.join(path, "policy_loss_{}".format(episode)))
        #     plt.close()
        #     p_loss = []
        #
        #     plt.title("value loss for {} episode".format(episode))
        #     plt.plot(v_loss)
        #     plt.xlabel("every 200 episode")
        #     plt.ylabel("every 200 episode average value loss")
        #     plt.savefig(os.path.join(path, "value_loss_{}".format(episode)))
        #     plt.close()
        #     v_loss = []
        #
        #     plt.title("reward for {} episode".format(episode))
        #     plt.plot(reward_record)
        #     plt.xlabel("every 200 episode")
        #     plt.ylabel("every 200 episode average reward")
        #     plt.savefig(os.path.join(path, "reward_record_{}".format(episode)))
        #     plt.close()
        #     reward_record = []
