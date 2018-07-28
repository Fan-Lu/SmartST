"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
# from maze_env import env
import config


class QLearningTable:
	def __init__(self, env, actions, learning_rate=config.learning_rate, reward_decay=config.reward_decay,
				 e_greedy=config.e_greedy):
		self.actions = actions  # a list
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
		self.env = env

		self.previous_action = None

	def choose_action(self, observation):	# todo modify valid action space
		self.check_state_exist(observation)
		# action selection
		# todo change to function
		# here I try to remove some actions, like going backward, going somewhere speed is 0, going out of board
		action_len = len(self.actions)
		action_init = list(range(action_len))
		# this is for e_greedy
		if np.random.uniform() < self.epsilon:
			# choose best action
			# print('=======choose best=======')
			state_action = self.q_table.loc[observation, :]
			state_action = state_action.reindex(np.random.permutation(state_action.index))	 # some actions have same value
			action = state_action.idxmax()
			# print('action is {}'.format(action))
		else:
			# choose random action
			# print(self.actions)
			# print('=======random=========')
			'''
			todo, not useful now
			'''
			# todo remove previous action ()
			# if self.previous_action is not None:
			# 	# print(self.previous_action)
			# 	if self.previous_action == 0:
			# 		action_remove = 1
			# 	elif self.previous_action == 1:
			# 		action_remove = 0
			# 	elif self.previous_action == 2:
			# 		action_remove = 3
			# 	else:  # self.previous_action == 3
			# 		action_remove = 2
			# action_init.remove(action_remove)

			# # remove out_of_board action
			# action_rest = []
			# for i in range(action_len - 1):
			# 	action = action_init[i]
			# 	s_, _, _ = self.env.step(action, flag_move=False)
			# 	# print('random choose action: {}'.format(action))
			# 	if is_out_of_board(s_):
			# 		# print('out of board')
			# 		# action_init.remove(action)
			# 		pass
			# 	# print('action_init becomes: {}'.format(action_init))
			# 	else:
			# 		action_rest.append(action)
			# print('init is')
			# print(action_init)
			action = np.random.choice(action_init)
			'''
			end
			'''

			# print('after remove, action is: {}'.format(action_init))
			# print('action is {}'.format(action))
			# print('----')
			# print('')
		self.previous_action = action
		action_chosen = action
		# print(self.q_table)
		return action_chosen

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		# a = str(a)
		# print('++++++++++++++++++++++')
		q_predict = self.q_table.loc[s, a]
		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
		else:
			q_target = r  # next state is terminal
		self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(
				pd.Series(
					[0]*len(self.actions),
					index=self.q_table.columns,
					name=state,
				)
			)

def get_valid_action(self, observation):	# todo not done
	total_action = [0, 1, 2, 3]
	valid_action = []
	for i in range(len(total_action)):
		this_action = total_action[i]
		s_, _, _ = env.step(this_action)
		if is_out_of_board(s_):
			pass
		else:
			valid_action.append(this_action)

	return None

def is_out_of_board(s_):
	# detect if out of board
	# True means out of board
	point_x = s_[0]
	point_y = s_[1]
	# print('s_ coords: {}'.format([point_x, point_y]))
	if isinstance(point_x, str):
		return False
	elif 5 <= point_x <= 365 and 5 <= point_y <= 365:
		return False
	else:
		return True

def is_zero_speed(s_):
	# decide whether speed is zeo or not
	# True means speed is 0
	speed_future = get_speed(s_, mat_speed_u)
	if speed_future == 0:
		return True
	else:
		return False
