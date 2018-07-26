import numpy as np
import pandas as pd

# from maze_env import Maze


# def get_valid_action(current_state, env):
#	 total_action = [0, 1, 2, 3]
#	 valid_action = []
#	 for i in range(len(total_action)):
#		 next_state, _, _ = env.step(valid_action[i])
#		 if

# todo move to env

def action_int2word(action_int):
	if action_int == 0:  # up
		return '上'
	elif action_int == 1:  # down
		return '下'
	elif action_int == 2:  # right
		return '右'
	elif action_int == 3:  # left
		return '左'

def save_q_table(q_table):
	q_table.to_csv('q_table.csv')

def load_q_table(file_dir, RL): # todo not correct & not neccessary
	# RL.q_table = pd.read_csv(file_dir, index_col=0, dtype={'0.0':np.float64, '1.0':np.float64, '2.0':np.float64, '3.0':np.float64})
	RL.q_table = pd.read_csv(file_dir, index_col=0)


# if __name__ == '__main__':
	# mat_speed_u = load_mat_speed('mat_txt')
	# env = Maze()