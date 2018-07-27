"""
Reinforcement learning maze example.

Red rectangle:		  explorer.
Black rectangles:	   hells	   [reward = -1].
Yellow bin circle:	  paradise	[reward = +1].
All other states:	   ground	  [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable
import time
import os
from utils import action_int2word, save_q_table, load_q_table
import config


def update(epoch):

	for episode in range(epoch):
		# initial observation
		print('==========episode {}/{}=========='.format(episode, epoch))

		total_step = 0
		action_list = []
		observation = env.reset()

		while True:
			# fresh env
			env.render()

			# RL choose action based on observation
			action = RL.choose_action(str(observation))
			action_list.append(action_int2word(action))

			# RL take action and get next observation and reward
			observation_, reward, done = env.step(action)
			# time.sleep(5)

			total_step += 1

			# RL learn from this transition
			RL.learn(str(observation), action, reward, str(observation_))

			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
			# if total_step == 5:
			# 	print('final reward is: {}'.format(reward))
				print('total step is: {}'.format(total_step))
				if len(action_list) < 20:
					print('action_list is: {}'.format(action_list)) #
				print('=======')
				# time.sleep(10)
				break

	# end of game
	print('game over')
	env.destroy()

if __name__ == "__main__":
	env = Maze()
	RL = QLearningTable(env=env, actions=list(range(env.n_actions)))
	# RL = QLearningTable(actions=['0', '1', '2', '3'])
	# try:
	# 	q_table_file = open('q_table.csv')
	# 	load_q_table(q_table_file, RL)
	# 	print('load q_table')
	# except FileNotFoundError:
	# 	print('no old q_table')
	# if os.path.exists('q_table.csv'):
	# 	load_q_table('q_table.csv', RL)
	# print(RL.q_table)
	env.after(500, update(config.epoch))


	env.mainloop()
	# print('run here')
	save_q_table(RL.q_table)
	print(RL.q_table)
	print('ok')
	# load_q_table('q_table.csv', RL)
	# print('ok2')