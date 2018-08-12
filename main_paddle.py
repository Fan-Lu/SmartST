from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np

import os
import sys
from Environment_V2 import environment

from Model import Actor, Critic


actor = Actor(A_DIM=8)
critic = Critic()

use_cuda = False  # set to True if training with GPU

ENV = environment.env([21, 14], [45, 87], 999)

action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
GAMMA = 0.99


if __name__ == '__main__':

    # define input data and variable

    value_point = ENV.data_base.value_point
    episode = 0
    actor.build_net()  # build net first
    critic.build_net()

    while True:

        current_state = np.array(ENV.reset(start_loc=value_point[15], target=[48, 46], time=1), dtype='float32')

        for step in range(10000):
            # get real action

            real_action = actor.act(current_state)

            next_state, reward, done, info = ENV.step(real_action)

            #   Critic Learn
            c_hx, c_cx = critic.get_values('c')
            td_error = critic.get_values('td_error')
            critic.train(current_state, next_state, reward, c_cx, c_hx, GAMMA)

            #   Actor Learn
            a_hx, a_cx = actor.get_values()
            actor.train(current_state, a_cx, a_hx, td_error)

            current_state = next_state

            print('Episode: {} Step: {} Aciton: {}'.format(episode, step, real_action))

            if done:
                episode += 1
                break




