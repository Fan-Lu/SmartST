
# reward
reward_zero_speed_current = -50		# in case it goes into zero-speed area
reward_zero_speed_future = -50
reward_target = 200
reward_step = -10		# in case it goes bachward	todo forbid it?
reward_out_board = -50	# in case it goes out-board

# parameter
alpha = 0.025	# used in reward function: reward = -alpha*distance + (1-alpha)*speed + reward_action
epoch = 2		# play epoch times

# parameter for RL
e_greedy=0.9
learning_rate = 0.01
reward_decay = 0.9

# destiny position
position = 6

# speed image
image_file = 'image_speed.jpeg'




