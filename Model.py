from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy
import sys

def conv_bn_layer(main_input, ch_out, filter_size, stride, padding, act='relu'):
    conv = fluid.layers.conv2d(
        input=main_input, # shape = [N,C,H,W]
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        use_cudnn=True,
        use_mkldnn=False,
        act=act)
    return conv

class Actor(object):
    def __init__(self, A_DIM=4):
        self.A_DIM = A_DIM
        self.prob = None
        self.a_hx = None
        self.a_cx = None
        self.prob_program =None

    def get_input(self):
        # create input
        s = fluid.layers.data(name='current_state', shape=[3, 100, 100], dtype='float32')
        a_cx = fluid.layers.data(name='a_cx', shape=[256], dtype='float32')
        a_hx = fluid.layers.data(name='a_hx', shape=[256], dtype='float32')
        td_error = fluid.layers.data(name='td_error', shape=[1], dtype='float32')

        return s, a_cx, a_hx, td_error

    def act(self, state):
        return self.exe.run(self.prob_program, feed={'state': state})

    def get_values(self):
        return self.a_hx, self.a_cx

    def build_net(self):
        s, a_cx, a_hx, td_error = self.get_input()
        a_conv1 = conv_bn_layer(main_input=s, ch_out=32, filter_size=3, stride=2, padding=1)
        a_conv2 = conv_bn_layer(main_input=a_conv1, ch_out=32, filter_size=3, stride=2, padding=1)
        a_conv3 = conv_bn_layer(main_input=a_conv2, ch_out=32, filter_size=3, stride=2, padding=1)
        a_conv4 = conv_bn_layer(main_input=a_conv3, ch_out=32, filter_size=3, stride=2, padding=1)
        a_fc5 = fluid.layers.fc(input=a_conv4, size=50, act='relu')
        self.a_hx, self.a_cx = fluid.layers.lstm_unit(x_t=a_fc5, hidden_t_prev=a_hx, cell_t_prev=a_cx)

        fc = fluid.layers.fc(input=self.a_hx, size=self.A_DIM)
        probs = fluid.layers.softmax(input=fc)
        self.prob_program = fluid.default_main_program().clone()
        lprobs = fluid.layers.log(probs)  # log operation 1*8

        log_prob = fluid.layers.reduce_max(fluid.layers.reduce_max(lprobs, dim=0))   # get probability at index action
        td_error = fluid.layers.reduce_max(td_error, dim=0)
        # print(td_error.shape, log_prob.shape)
        # neg_log_prob = paddle.fluid.layers.mul(log_prob, td_error)
        # print(td_error.shape, log_prob.shape)
        # neg_log_prob = log_prob * td_error * -1.0
        neg_log_prob = log_prob

        # define optimizer
        optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
        optimizer.minimize(neg_log_prob)

        # define program
        self.train_program_actor = fluid.default_main_program()

        # fluid exe
        place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)
        self.exe.run(fluid.default_startup_program())

    def train(self, state, a_cx, a_hx, td_error):
        self.exe.run(self.train_program_actor,
                     feed={
                         'current_state': state,
                         'a_cx': a_cx,
                         'a_hx': a_hx,
                         'td_error': td_error
                     })

class Critic(object):
    def __init__(self):
        self.c_cx = None
        self.c_hx = None
        self.td_error = None

    def get_input(self):
        # create input
        s = fluid.layers.data(name='current_state', shape=[3, 100, 100])
        s_ = fluid.layers.data(name='next_state', shape=[3, 100, 100])
        reward = fluid.layers.data(name='reward', shape=[1])
        self.c_cx = fluid.layers.data(name='c_cx', shape=[256])
        self.c_hx = fluid.layers.data(name='c_hx', shape=[256])
        gamma = fluid.layers.data(name='gamma', shape=[1])
        return s, s_, reward, gamma

    def get_values(self, para):
        if para == 'c':
            return self.c_hx, self.c_cx
        elif para == 'td_error':
            return self.td_error
        else:
            raise ['No matching parameters!']


    def build_net(self):
        s, s_, reward, gamma= self.get_input()
        v_curr, c_hx, c_cx = self.predict_values(s, self.c_hx, self.c_cx)
        v_next, self.c_hx, self.c_cx = self.predict_values(s_, c_hx, c_cx)
        self.td_error = reward + gamma * v_next - v_curr

        # define optimizer
        optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
        optimizer.minimize(self.td_error)

        # define program
        self.train_program_critic = fluid.default_main_program()

        # fluid exe
        place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)
        self.exe.run(fluid.default_startup_program())

    def predict_values(self, s, c_hx, c_cx):
        a_conv1 = conv_bn_layer(main_input=s, ch_out=32, filter_size=3, stride=2, padding=1)
        a_conv2 = conv_bn_layer(main_input=a_conv1, ch_out=32, filter_size=3, stride=2, padding=1)
        a_conv3 = conv_bn_layer(main_input=a_conv2, ch_out=32, filter_size=3, stride=2, padding=1)
        a_conv4 = conv_bn_layer(main_input=a_conv3, ch_out=32, filter_size=3, stride=2, padding=1)
        a_fc5 = fluid.layers.fc(input=a_conv4, size=256, act='relu')

        hx, cx = fluid.layers.lstm_unit(x_t=a_fc5, hidden_t_prev=c_hx, cell_t_prev=c_cx)
        state = hx
        fc = fluid.layers.fc(input=state, size=1)
        value = fluid.layers.softmax(input=fc)
        return value, cx, hx

    def train(self, current_state, next_state, reward, c_cx, c_hx, gamma):
        self.exe.run(self.train_program_critic,
                     feed={
                         'current_state': current_state,
                         'next_state': next_state,
                         'reward': reward,
                         'c_cx': c_cx,
                         'c_hx': c_hx,
                         'gamma': gamma}
                     )