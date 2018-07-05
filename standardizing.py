import numpy as np
import os

dire = 'origin'

raw_dir = os.path.join(os.getcwd(), dire)
file_dir = os.listdir(raw_dir)
file_dir = [os.path.join(raw_dir, name) for name in file_dir]
max_number_0 = 0
max_number_1 = 0
for fil in file_dir:
	tmp = np.load(fil)
	tmp_max_0 = np.max(tmp[:,:,:,0])
	tmp_max_1 = np.max(tmp[:,:,:,1])
	if tmp_max_0 > max_number_0:
		max_number_0 = tmp_max_0
	if tmp_max_1 > max_number_1:
		max_number_1 = tmp_max_1

for fil in file_dir:
	tmp = np.load(fil)
	tmp[:,:,:,0] = tmp[:,:,:,0] / max_number_0
	tmp[:,:,:,1] = tmp[:,:,:,1] / max_number_1
	np.save(fil[-12:-4]+'(normalized)',tmp)
	
print("0 channel's maximum number:",max_number_0)
print("1 channel's maximum number:",max_number_1)
