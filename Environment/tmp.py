import os
import numpy as np


class average:
    def __init__(self):
        self.padding()
        self.sum_average()
        self.add_average()

    def sum_average(self):
        sum = np.zeros([24, 100, 100, 2])
        for i in range(20161101, 20161131):
            tmp = getattr(self, str(i))[1]
            for j in range(tmp.shape[0]):
                for k in range(self.mask.__len__()):
                    a = self.mask[k][0]
                    b = self.mask[k][1]
                    if tmp[j, a, b]!=0:
                        sum[j, a, b, 0] += tmp[j, a, b]
                        sum[j, a, b, 1] += 1
        new_sum = sum[:, :, :, 0] / (sum[:, :, :, 1] + 1e-7)
        setattr(self, "sum_aver", new_sum)

    def add_average(self):
        tmp_sum = getattr(self, "sum_aver")
        for xx in range(20161101, 20161131):
            tmp = getattr(self, str(xx))[0]
            tmp_1 = getattr(self, str(xx))[1]
            for i in range(tmp.shape[0]):
                for j in range(self.mask.__len__()):
                    a = self.mask[j][0]
                    b = self.mask[j][1]
                    if tmp[i, a, b] == 0 and tmp_1[int(i/24), a, b] != 0:
                        tmp[i, a, b] = tmp_1[int(i/24), a, b]
                    else :
                        tmp[i, a, b] = tmp_sum[int(i/24), a, b]
            np.save(str(xx) + "add_with_average", tmp)

    def padding(self):
        tmp = os.listdir(os.getcwd())
        names = [os.path.join(os.getcwd(), name) for name in tmp]
        for i in names:
            if i.endswith('mask.npy'):
                tmp = np.load(i)
                tmp = tmp[99:199, 99:199]
                value_point = []
                for i in range(tmp.shape[0]):
                    for j in range(tmp.shape[1]):
                        if tmp[i, j] == 1:
                            value_point.append([i,j])
                setattr(self, 'mask', value_point)

        for i in names:
            if os.path.isfile(i) and i.endswith('.npy') and not i.endswith('mask.npy'):
                tmp = np.load(i)
                tmp1 = self.yasuo(tmp, i)
                setattr(self, i[-12:-4], [tmp, tmp1])

    def yasuo(self, array, name):
        tmp = np.zeros([24, 100, 100, 2])
        for i in range(array.shape[0]):
            for j in range(self.mask.__len__()):
                a = self.mask[j][0]
                b = self.mask[j][1]
                try:
                    if array[i, a, b] != 0:
                        tmp[int(i/24), a, b, 0] += array[i, a, b]
                        tmp[int(i/24), a, b, 1] += 1
                except:
                    print("[{},{},{}] in ".format(i,a,b),name)
        tmp_1 = tmp[:, :, :, 0] / (tmp[:, :, :, 1]+1e-9)
        return tmp_1

a = average()