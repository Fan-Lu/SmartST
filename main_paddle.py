from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np

import os
import sys
from Environment_V2 import environment

from Model_actor_critic import Actor, Critic, AC

#
# data = fluid.layers.data(name='X', shape=[1], dtype='float32')
# hidden = fluid.layers.fc(input=data, size=10)
#
# loss = fluid.layers.mean(hidden)
# adam = fluid.optimizer.Adam()
# adam.minimize(loss)
# cpu = fluid.core.CPUPlace()
# exe = fluid.Executor(cpu)
# exe.run(fluid.default_startup_program())
# x = np.random.random(size=(10, 1)).astype('float32')
# outs = exe.run(
#             feed={'X': x},
#             fetch_list=[loss.name])
# print(outs)
#
# outs1 = exe.run(
#             feed={'X': x},
#             fetch_list=[hidden.name])
# print(outs1)

use_cuda = False  # set to True if training with GPU

ENV = environment.env([21, 14], [45, 87], 999)

action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
GAMMA = 0.99

# actor = Actor(A_DIM=8)
# critic = Critic(GAMMA)
ac = AC(A_DIM=8, gamma=GAMMA)

if __name__ == '__main__':

    # define input data and variable

    value_point = ENV.data_base.value_point
    episode = 0
    # actor.build_net()  # build net first
    # critic.build_net()
    ac.build_net()

    while True:

        current_state = ENV.reset(start_loc=value_point[15], target=[48, 46], time=1)

        for step in range(10000):

            action_pros = ac.act(current_state)  # get real action
            action_list = np.random.multinomial(20, list(action_pros[0]*0.95), size=1)  # generate distribution
            real_action = action_dic[int(np.argmax(action_list))]  # get real action

            next_state, reward, done, info, success = ENV.step(real_action)

            print("reward is {0}".format(reward))

            #   Critic Learn
            td_error = ac.get_td_error(current_state, next_state, reward)
            print("get td")

            #   Critic Learn
            td_error = ac.get_td_error(current_state, next_state, reward)
            ac.train('c', current_state=current_state, next_state=next_state, reward=reward, td_error=td_error)
            print("critic train!")

            #   Actor Learn
            ac.train('a', current_state=current_state, next_state=next_state, reward=reward, td_error=td_error)
            print("actor train!")

            current_state = next_state

            print('Episode: {} Step: {} Aciton: {}'.format(episode, step, real_action))

            if done:
                episode += 1
                break




