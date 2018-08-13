from __future__ import print_function
import numpy as np
from Environment_V2 import environment
from Model_dpg import PolicyGradient

use_cuda = False  # set to True if training with GPU

ENV = environment.env([21, 14], [45, 87], 999)

action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
GAMMA = 0.99


# args = GetConfiguration()
# args.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) + '/SmartST/model_saved_rl/'
# args.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) + '/SmartST/result_saved_rl/'

PG = PolicyGradient(A_DIM=8, lr=0.001, reward_decay=GAMMA)
value_point = ENV.data_base.value_point

if __name__ == '__main__':

    episode = 0
    PG.build_net()

    while True:
        current_state = np.array(ENV.reset(start_loc=value_point[15], target=[48, 46], time=1), dtype='float32')
        print(current_state.shape)
        step = 0

        for step in range(10000):
            # get action
            action_pros = PG.act(current_state)
            action_list = np.random.multinomial(20, list(action_pros[0]*0.95), size=1)  # generate distribution
            real_action = action_dic[int(np.argmax(action_list))]  # get real action

            next_state, reward, done, info, success = ENV.step(real_action)
            next_state = np.array(next_state, dtype='float32')
            print("reward is {0}".format(reward))

            step += 1

            # print('Episode: {} Step: {} Aciton: {}'.format(episode, step, action_dic[int(real_action)]))

            if done:

                PG.train(current_state, reward)
                episode += 1
                break

            current_state = next_state