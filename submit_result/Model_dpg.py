"""
Policy Gradient
"""


from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import sys

def conv_bn_layer(main_input, ch_out, filter_size, stride, padding, act=None, name=None):
    conv = fluid.layers.conv2d(
        input=main_input,  # shape = [N,C,H,W]
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        use_cudnn=True,
        act=act,
        name=name
    )
    return conv


def bn(x, name=None, act='sigmoid'):
    return fluid.layers.batch_norm(x, name=name, act=act)


class PolicyGradient(object):

    def __init__(self, A_DIM=4, lr=0.001, reward_decay=0.94):
        self.A_DIM = A_DIM
        self.probs = None
        # self.inferece_program = None
        self.lr = lr
        self.state_record, self.action_record, self.reward_record = [], [], []  # lists used to store trainsitions
        self.gamma = reward_decay
        self.inferece_program = None

        place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)

    def get_input(self):
        # create input
        state = fluid.layers.data(name='current_state', shape=[3, 50, 50], dtype='float32')
        action = fluid.layers.data(name='action', shape=[1], dtype='int64')
        reward = fluid.layers.data(name='reward', shape=[1], dtype='float32')
        return state, action, reward

    def act(self, state):
        # state = np.expand_dims(state, axis=0)
        # print(state.shape)
        return self.exe.run(self.inferece_program,
                            feed={'current_state': state},
                            fetch_list=[self.probs])[0]

    def network(self, s):
        a_conv1 = conv_bn_layer(main_input=s, ch_out=32, filter_size=3, stride=2, padding=1, name='a_conv1')
        a_conv1_bn = bn(a_conv1, name='a_conv1_bn')
        a_conv2 = conv_bn_layer(main_input=a_conv1_bn, ch_out=32, filter_size=3, stride=2, padding=1, name='a_conv2')
        a_conv2_bn = bn(a_conv2, name='a_conv2_bn')
        a_conv3 = conv_bn_layer(main_input=a_conv2_bn, ch_out=32, filter_size=3, stride=2, padding=0, name='a_conv3')
        a_conv3_bn = bn(a_conv3, name='a_conv3_bn')
        a_conv4 = conv_bn_layer(main_input=a_conv3_bn, ch_out=32, filter_size=3, stride=1, padding=0, name='a_conv4')
        a_conv4_bn = bn(a_conv4, name='a_conv4_bn')
        a_fc5 = fluid.layers.fc(input=a_conv4_bn, size=32, act=None, name='a_fc5')
        a_fc6 = fluid.layers.fc(input=a_fc5, size=16, act=None, name='a_fc6')
        return fluid.layers.fc(input=a_fc6, size=self.A_DIM, act='softmax', name='a_out')

    def build_net(self):
        s, action, reward = self.get_input()
        self.probs = self.network(s)
        self.inferece_program = fluid.default_main_program().clone()  # 1*8

        neg_log_prob = fluid.layers.cross_entropy(
            input=self.probs,
            label=action)
        neg_log_prob_weight = fluid.layers.elementwise_mul(x=neg_log_prob, y=reward)
        loss = fluid.layers.reduce_mean(
            neg_log_prob_weight)  # reward guided loss

        # neg_log_prob = fluid.layers.reduce_mean(log_prob * reward * -1.0)

        # define optimizer
        optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
        optimizer.minimize(loss)

        # define program
        # self.train_program_actor = fluid.default_main_program()

        # fluid exe
        self.exe.run(fluid.default_startup_program())

    def train(self):
        discounted_reward = self.discount_and_norm_rewards()
        state_record = np.vstack(self.state_record).astype("float32")
        action_record = np.array(self.action_record).astype("int64").reshape([-1, 1])
        reward_record = discounted_reward.astype("float32").reshape([-1, 1])

        self.exe.run(fluid.default_main_program(),
                     feed={
                         'current_state': state_record,
                         'action': action_record,
                         'reward': reward_record
                     })
        self.state_record, self.action_record, self.reward_record = [], [], []

    def store_transition(self, state, action, reward):
        self.state_record = state
        self.action_record = action
        self.reward_record = reward

        # return self.obs, self.ep_as, self.ep_rs, self.ep_pbs

    def discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.reward_record)
        running_add = 0
        for t in reversed(range(len(self.reward_record))):
            running_add = running_add * self.gamma + self.reward_record[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
