import os
import numpy as np

name = os.getcwd()
a = os.listdir(os.getcwd())
files = [os.path.join(name, names) for names in a]
result = np.zeros([100, 100, 2])
for file in files:
    if os.path.isfile(file) and file.endswith('.npy') and not file.endswith('maskV1.npy'):
        tmp = np.load(file)
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                for k in range(tmp.shape[2]):
                    if tmp[i, j, k] != 0:
                        result[j, k, 0] += tmp[i, j, k]
                        result[j, k, 1] += 1
    if os.path.isfile(file) and file.endswith('maskV1.npy'):
        mask = np.load(file)
true_result = result[:, :, 0] / (result[:, :, 1]+1e-7)
true_result = true_result * mask
np.save("static", true_result)