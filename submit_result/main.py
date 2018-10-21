from __future__ import print_function
import numpy as np
from environment_static_2 import environment
from Model_dpg import PolicyGradient

use_cuda = False  # set to True if training with GPU

maps = ["map_62.npy"]
games = {"map_62.npy": [[102, 292]]}
use_running_reward = True

ENV = environment(maps=maps, map_size=(50, 50), games=games, only_when_success=True, digital=False, reward_type="one", way_back=False, running_reward=use_running_reward,
                 running_reward_interval=10)

action_dic = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
GAMMA = 0.99
max_time = 30

# args = GetConfiguration()
# args.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) + '/SmartST/model_saved_rl/'
# args.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) + '/SmartST/result_saved_rl/'

Agent = PolicyGradient(A_DIM=8, lr=0.001, reward_decay=GAMMA)

if __name__ == '__main__':

    success_counter = 0
    max_episode = 100
    Agent.build_net()

    for episode in range(max_episode):
        current_state, _, mask = ENV.reset(plot=True)
        current_state = np.array(current_state)[np.newaxis, :].astype("float32")

        state_record = []
        reward_record = []
        action_record = []
        final_reward = 0

        for step in range(max_time):
            # get action
            action_probs = Agent.act(current_state)

            state_record.append(current_state)

            action_probs = np.array(action_probs[0]) * mask
            action = np.random.choice(range(action_probs.shape[0]), p=action_probs.ravel() / np.sum(action_probs.ravel()))  # generate distribution
            real_action = action_dic[action]

            if step == max_time - 1:
                next_state, location_information, mask, r, success, f_r, running_mean_reward = ENV.step(real_action, last_step=True)
            else:
                next_state, location_information, mask, r, success, f_r, running_mean_reward = ENV.step(real_action)

            if (step == max_time - 1) or success:
                if running_mean_reward and use_running_reward:
                    final_reward = f_r - running_mean_reward
                else:
                    final_reward = f_r


            reward_record.append(r)
            action_record.append(action)

            if success:
                success_counter += 1
                break

            current_state = np.array(next_state)[np.newaxis, :].astype("float32")

        for i in range(len(reward_record)):
            reward_record[i] = reward_record[i] + final_reward

        Agent.store_transition(state=state_record, action=action_record, reward=reward_record)
        Agent.train()
        print("Episode:", episode, "success time:", success_counter)

            # print('Episode: {} Step: {} Aciton: {}'.format(episode, step, action_dic[int(real_action)]))