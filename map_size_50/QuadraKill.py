import os
import sys
from Environment_V2 import environment_static as environment
import torch
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

from Agent.ac import Actor, Critic

tra_num = 5
# actor = Actor(A_DIM=8).cuda()
# critic = Critic().cuda()

actor = Actor(A_DIM=8)
critic = Critic()

success_counter = 0

# Normalize([3.1405, 0.584, 0.584], [5.1829, 4.2307, 4.2307])

def normalize(s):
    s[0] = s[0]/20
    s[1] = s[1]/50
    # s[2] = s[2]/50
    return s

a_opt = optim.Adam(actor.parameters(), lr=0.0005)
c_opt = optim.Adam(critic.parameters(), lr=0.0005)

ENV = environment.env([1,1],[2,2], 999, plot=True)

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

    value_point = ENV.data_base.value_point
    length = len(value_point)
    data_pool = (np.random.random([2, tra_num])*length).astype(int)
    episode = 0
    np.random.seed(1)

    p_loss = []
    v_loss = []
    reward_record = []

    p_loss_tmp = []
    v_loss_tmp = []
    reward_record_tmp = []

    while True:

        a = np.random.randint(0, tra_num)
        Time = np.random.randint(0, 10000)
        if data_pool[0,a] == data_pool[1,a]:
            print('a equals b, this is ridiculous')
            continue
        s, valid_action = ENV.reset(start_loc=value_point[data_pool[0, a]], target=value_point[data_pool[1, a]], time=10)
        s = normalize(s)
        if np.sum(valid_action) == 0:
            print("shibai")
            continue
        print("Episode:{} ".format(episode),"Index:",value_point[data_pool[0,a]],' ',value_point[data_pool[1,a]],"success times:{}".format(success_counter))
        # a_cx = Variable(torch.Tensor(1, 64)).cuda()
        # a_hx = Variable(torch.Tensor(1, 64)).cuda()
        #
        # c_cx = Variable(torch.Tensor(1, 64)).cuda()
        # c_hx = Variable(torch.Tensor(1, 64)).cuda()

        a_cx = Variable(torch.Tensor(1, 64))
        a_hx = Variable(torch.Tensor(1, 64))

        c_cx = Variable(torch.Tensor(1, 64))
        c_hx = Variable(torch.Tensor(1, 64))

        all_rewards = []
        all_values = []
        all_entropies = []
        all_lprobs = []

        for step in range(max_times):
            try:
                s = Variable(torch.from_numpy(np.array(s))).view(1, 3, 50, 50).float()
                tmp1 = s[0, 0, :, :]
                tmp2 = s[0, 1, :, :]
            except:
                print("stop here")
            value, (c_hx, c_cx) = critic((s, (c_hx, c_cx)))
            probs, (a_hx, a_cx) = actor((s, (a_hx, a_cx)))

            # mask = Variable(torch.from_numpy(valid_action)).cuda().view(1, 8)
            mask = Variable(torch.from_numpy(valid_action)).view(1, 8)
            masked_probs = probs * mask

            try:
                action = masked_probs.multinomial(1)
            except:
                print("stop here")
            lporbs = torch.log(probs)
            log_prob = lporbs.gather(1, action)
            entropy = -(log_prob * probs).sum(1)
            try:
                real_action = action_dic[int(action.cpu().data.numpy())]
            except:
                print("stop here")
            s_, r, done, [_, _, valid_action], success = ENV.step(real_action)   # True: Read terminal
            if np.sum(valid_action) == 0: # used to deal with the environment's dirty data.
                break

            all_rewards.append(r)
            all_values.append(value)
            all_entropies.append(entropy)
            all_lprobs.append(log_prob)

            s = s_
            s = normalize(s)
            # print('Episode: {} Step: {} Aciton: {}'.format(episode, step, real_action))

            # if success:
            #     if not os.path.exists('/home/exx/Lab/SmartST/model_saved_rl'):
            #         os.mkdir('/home/exx/Lab/SmartST/model_saved_rl')
            #     torch.save(actor.state_dict(), '/home/exx/Lab/SmartST/model_saved_rl/' + 'suc_model_{:d}.pth'.format(episode))
            if success:
                success_counter += 1

            if done or step == max_times-1:
                episode += 1
                break

        # R = Variable(torch.zeros(1, 1)).cuda()
        R = Variable(torch.zeros(1, 1))
        if not done:
            try:
                # s = Variable(torch.from_numpy(np.array(s))).view(1, 3, 50, 50).float().cuda()
                s = Variable(torch.from_numpy(np.array(s))).view(1, 2, 50, 50).float()
            except:
                print("stop here")
            value, (_, _) = critic((s, (c_hx, c_cx)))
            R = value
        all_values.append(R)
        policy_loss = 0
        value_loss = 0
        # gae = Variable(torch.zeros(1, 1)).cuda()
        gae = Variable(torch.zeros(1, 1))
        for i in reversed(range(len(all_rewards))):
            R = GAMMA * R + all_rewards[i]
            advantage = R - all_values[i]
            value_loss += 0.5 * advantage.pow(2)

            # a = all_values[i+1]
            #   Generalized Advantage Estimation
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

        p_loss_tmp.append(policy_loss.data)
        v_loss_tmp.append(value_loss.data)
        reward_record_tmp.append(sum(all_rewards)/all_rewards.__len__())

        if (episode+1) % 200 == 0:
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


        if (episode + 1) % 50000 == 0:
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
