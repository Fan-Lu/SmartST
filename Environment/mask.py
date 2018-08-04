import numpy as np
import matplotlib.pyplot as plt
import csv
#
# counter = 0
# tmp = np.zeros([200, 200])
# for i in range(20161107, 20161131):
#     name = str(i) + '_counter.npy'
#     tmp1 = np.load(name)
#     tmp1 -= 1e-7
#     for j in range(288):
#         tmp += tmp1[j, :, :]
#         counter += 1
# # tmp /= counter
# print("total trajectory accounted is :{}".format(counter))
# np.save('mask_jishu', tmp)
#
# liebiao = []
# for i in range(200):
#     for j in range(200):
#         liebiao.append(float(tmp[i, j]))
#
# liebiao.sort()
# with open('mask.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     for i in range(len(liebiao)):
#         writer.writerow([liebiao[i]])
#
# plt.plot(liebiao)
# plt.title('shujufenbu')
# plt.savefig('shujutongji.PNG')
# plt.show()

# a = np.load('mask_jishu.npy')
# result = np.array(a>2000,dtype=int)
# np.save('mask', result)


# mask = np.load('mask.npy')
#
# for i in range(20161101, 20161131):
#     name = str(i) + '.npy'
#     tmp = np.load(name)
#     tmp = tmp * mask
#     np.save(str(i), tmp)


# for i in range(20161101, 20161131):
#     name = str(i) + '.npy'
#     tmp = np.load(name)
#     tmp = tmp[:, 99:199, 99:199]
#     np.save(str(i), tmp)
