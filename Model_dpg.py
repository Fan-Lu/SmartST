"""
Policy Gradient
"""


from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import sys

def conv_bn_layer(main_input, ch_out, filter_size, stride, padding, act='relu'):
    conv = fluid.layers.conv2d(
        input=main_input,  # shape = [N,C,H,W]
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        use_cudnn=True,
        use_mkldnn=False,
        act=act
    )
    return conv

class PolicyGradient(object):

    def __init__(self, A_DIM=4, lr=0.001, reward_decay=0.94):
        self.A_DIM = A_DIM
        self.probs = None
        self.log_prob = None
        self.prob_program = None
        self.lr = lr
        self.obs, self.ep_as, self.ep_rs, self.ep_pbs = [], [], [], []  # lists used to store trainsitions
        self.gamma = reward_decay

        place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)

    def get_input(self):
        # create input
        s = fluid.layers.data(name='current_state', shape=[3, 100, 100], dtype='float32')
        reward = fluid.layers.data(name='reward', shape=[1], dtype='float32')
        return s, reward

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        print(state.shape)
        return self.exe.run(self.prob_program,
                            feed={'current_state': state},
                            fetch_list=[self.probs])[0]

    def build_net(self):
        s, reward = self.get_input()
        a_conv1 = conv_bn_layer(main_input=s, ch_out=32, filter_size=3, stride=2, padding=1)
        a_conv2 = conv_bn_layer(main_input=a_conv1, ch_out=32, filter_size=3, stride=2, padding=1)
        a_conv3 = conv_bn_layer(main_input=a_conv2, ch_out=32, filter_size=3, stride=2, padding=1)
        a_conv4 = conv_bn_layer(main_input=a_conv3, ch_out=32, filter_size=3, stride=2, padding=1)
        a_fc5 = fluid.layers.fc(input=a_conv4, size=50, act='relu')

        self.probs = fluid.layers.fc(input=a_fc5, size=self.A_DIM, act='softmax')
        self.prob_program = fluid.default_main_program().clone()  # 1*8

        lprobs = fluid.layers.log(self.probs)  # log operation 1*8
        log_prob = fluid.layers.reduce_max(lprobs, dim=1, keep_dim=True)

        neg_log_prob = fluid.layers.reduce_mean(log_prob * reward * -1.0)

        # define optimizer
        optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
        optimizer.minimize(neg_log_prob)

        # define program
        self.train_program_actor = fluid.default_main_program()

        # fluid exe
        self.exe.run(fluid.default_startup_program())

    def train(self, current_state, reward):
        current_state = np.expand_dims(current_state, axis=0)
        print(current_state.shape)
        reward = np.expand_dims(np.array([reward], dtype='float32'), -1)
        print(reward)
        self.exe.run(self.train_program_actor,
                     feed={
                         'current_state': current_state,
                         'reward': reward
                     })

    def store_transition(self, s, a, r, probs):
        self.obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
        self.ep_pbs.append(probs)

        return self.obs, self.ep_as, self.ep_rs, self.ep_pbs

    def discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
