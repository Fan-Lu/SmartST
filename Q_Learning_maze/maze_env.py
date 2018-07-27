"""
Reinforcement learning maze example.

Red rectangle:		  explorer.
Black rectangles:	   hells	   [reward = -1].
Yellow bin circle:	  paradise	[reward = +1].
All other states:	   ground	  [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
	import Tkinter as tk
else:
	import tkinter as tk

# from utils import load_mat_speed, get_speed
import config

UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width


class Maze(tk.Tk, object):
	def __init__(self):
		super(Maze, self).__init__()
		self.action_space = ['u', 'd', 'l', 'r']
		self.n_actions = len(self.action_space)
		self.title('maze')
		self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
		self._build_maze()

		self.mat_speed = load_mat_speed('mat.txt')
		# print(self.mat_speed)
		# print('d')

	def _build_maze(self):
		self.canvas = tk.Canvas(self, bg='white',
						   height=MAZE_H * UNIT,
						   width=MAZE_W * UNIT)

		# create grids
		for c in range(0, MAZE_W * UNIT, UNIT):
			x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
			self.canvas.create_line(x0, y0, x1, y1)
		for r in range(0, MAZE_H * UNIT, UNIT):
			x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
			self.canvas.create_line(x0, y0, x1, y1)

		# create origin
		origin = np.array([20, 20])

		# # hell (we don't need hell
		# hell1_center = origin + np.array([UNIT * 2, UNIT])
		# self.hell1 = self.canvas.create_rectangle(
		#	 hell1_center[0] - 15, hell1_center[1] - 15,
		#	 hell1_center[0] + 15, hell1_center[1] + 15,
		#	 fill='black')
		# # hell
		# hell2_center = origin + np.array([UNIT, UNIT * 2])
		# self.hell2 = self.canvas.create_rectangle(
		#	 hell2_center[0] - 15, hell2_center[1] - 15,
		#	 hell2_center[0] + 15, hell2_center[1] + 15,
		#	 fill='black')

		# create oval
		oval_center = origin + UNIT * config.position
		self.oval = self.canvas.create_oval(
			oval_center[0] - 15, oval_center[1] - 15,
			oval_center[0] + 15, oval_center[1] + 15,
			fill='yellow')

		# create red rect
		self.rect = self.canvas.create_rectangle(
			origin[0] - 15, origin[1] - 15,
			origin[0] + 15, origin[1] + 15,
			fill='red')

		# pack all
		self.canvas.pack()

	def reset(self):
		self.update()
		time.sleep(0.5)
		self.canvas.delete(self.rect)
		origin = np.array([20, 20])
		self.rect = self.canvas.create_rectangle(
			origin[0] - 15, origin[1] - 15,
			origin[0] + 15, origin[1] + 15,
			fill='red')
		# return observation
		return self.canvas.coords(self.rect)

	def step(self, action, flag_move=True):
		reward_action = 0	# reward for action (reach target; cost of one step; enter 0 speed block)
		s = self.canvas.coords(self.rect)
		# print(s)	# [point_1_horizontal, point_1_vertical, point_2_x, point_2_y]
		# time.sleep(5)
		base_action = np.array([0, 0])
		if action == 0:   # up
			if s[1] > UNIT:	# todo use valid action
				base_action[1] -= UNIT
			else:
				reward_action += config.reward_out_board
		elif action == 1:   # down
			if s[1] < (MAZE_H - 1) * UNIT:
				base_action[1] += UNIT
			else:
				reward_action += config.reward_out_board
		elif action == 2:   # right
			if s[0] < (MAZE_W - 1) * UNIT:
				base_action[0] += UNIT
			else:
				reward_action += config.reward_out_board
		elif action == 3:   # left
			if s[0] > UNIT:
				base_action[0] -= UNIT
			else:
				reward_action += config.reward_out_board
		# print(base_action)
		# time.sleep(5)
		self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

		s_ = self.canvas.coords(self.rect)  # next state

		if not flag_move:
			self.canvas.move(self.rect, -base_action[0], -base_action[1])
		# reward function
		if s_ == self.canvas.coords(self.oval):
			reward_action += config.reward_target
			done = True
			s_ = 'terminal'
			return s_, reward_action, done
		# elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
		#	 reward = -1
		#	 done = True
		#	 s_ = 'terminal'
		else:
			reward_action += config.reward_step
			done = False

		'''
		change reward function:
		1. time
		2. distance
		'''
		# distance (Euclidean Distance )
		des = self.canvas.coords(self.oval) # [point_1_x, point_1_y, point_2_x, point_2_y]
											# point_1 is upper left point of rectangle, point_2 is down right
		distance = np.sqrt((s[0] - des[0])**2 + (s[1] - des[1])**2)

		# speed (from coords we can refer to speed matrix)
		speed_current = get_speed(s, self.mat_speed)
		speed_next = get_speed(s_, self.mat_speed)
		reward_speed = (speed_current + speed_next) / 2	# use mean of current speed & next state's speed
		if speed_next == 0:	# try to avoid to go where speed is 0
			reward_action += config.reward_zero_speed_future
		elif speed_current == 0:
			reward_action += config.reward_zero_speed_current

		# time.sleep(0.5)
		# reward function todo reward is negtive
		alpha = config.alpha
		reward = (-1)*alpha*distance + (1-alpha)*reward_speed + reward_action
		# print('reward is: {}'.format(reward))
		# print('================')
		'''
		end
		'''

		return s_, reward, done

	def render(self):
		time.sleep(0.1)
		self.update()

def get_speed(state_current, mat_speed):
	# get speed from speed matrix
	# coords position -> index in matrix
	index_col = int(state_current[1] // 40)		# '/40' because each unit is 40 long
	index_row = int(state_current[0] // 40)
	# print('check current_coords: {}'.format([state_current[1], state_current[0]]))
	# print('check index: {}'.format([index_row, index_col]))
	if 0 <= index_row <= 9 and 0 <= index_col <= 9:
		speed = mat_speed[index_row][index_col]
	else:	# set speed = 0 if out of matrix
		speed = 0
	# print('check speed: {}'.format(mat_speed))
	return speed
# def reward_func(s_):
#
# 	return
def load_mat_speed(file_dir):
	# load speed matrix. todo later we will load from ?
	mat_speed = []
	with open(file_dir, 'r') as f:
		for row in f.readlines():
			row = row.strip().split()
			row = list(map(int, row))
			# print(row)
			# print(np.array(row))
			mat_speed.append(row)
	# print(mat)
	mat_speed = np.array(mat_speed)
	return mat_speed

def update():
	for t in range(10):
		s = env.reset()
		while True:
			env.render()
			a = 0
			s, r, done = env.step(a)
			if done:
				break

if __name__ == '__main__':
	env = Maze()
	env.after(100, update)
	env.mainloop()