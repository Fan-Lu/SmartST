import os
import numpy as np
import csv

class enviroment(): 
    def __init__(self, dir_name):
        self.load_files(dir_name)

    def load_files(self, dir_name):
        name = os.path.join(os.getcwd(), dir_name)
        a = os.listdir(os.getcwd())
        files = [os.path.join(os.getcwd(), name) for name in a]
        for file_path in files:
            if os.path.isfile(file_path) and file_path.endswith('.npy'):
                tmp = np.load(file_path)
                setattr(self, file_path[-18:-10], tmp)

        record = getattr(self, "20161101")
        for i in range(20161102, 20161131):
            tmp = getattr(self, '{}'.format(i))
            record = np.concatenate((record, tmp), axis=0)
            delattr(self, '{}'.format(i))
        setattr(self, "database", record)

    def geinitupian(self, time):
        if time > 2592000:
            raise ValueError('time must be within 30 days.')
        database = self.database
        yushu = time % 300
        index = int(time / 300)
        jiashu1 = database[index, :, :]
        jiashu2 = database[index+1, :, :]
        result = (yushu / 300) * jiashu2 + (1 - (yushu / 300)) * jiashu1
        return result

    def dayin(self, array, name='tmp_file.txt'):
        assert array.shape.__len__() == 2, 'input of array must has shape of two'
        # path = os.path.join(os.getcwd(), name)
        # if not os.path.exists(path):
        f = open(name, "w+")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                tmp = '{0:7}'.format(array[i,j]) + '\0'
                f.write(tmp)
            f.write('\n')


# a = enviroment('database')
# a.geinitupian(57)
# a.dayin(a.geinitupian(57))
#
# def save_fig(array, name, format=None):
#     array = array.numpy()
#     array *= 255
#     img = Image.fromarray(array, mode='L')
#     img.show()

def shuchuCSV(data_array, file_name='tmp'):
    assert type(data_array) == np.ndarray and data_array.shape.__len__() == 2, "Input must be a two-dimension array"
    with open(file_name + '.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(data_array.shape[0]):
            writer.writerows(data_array[0])

global enviroment_example
enviroment_example = enviroment('database')

class env:
    action_space = {'shang': [0, 1], 'you': [1, 0], 'xia': [0, -1], 'zuo': [-1, 0]}
    def __init__(self, start_loc, target, time, alpha = 0.5):
        self.start = start_loc
        self.target = target
        self.time = time
        self.data_base = enviroment_example
        self.observation = [self.data_base.geinitupian(time), self.target_and_loc(target), self.target_and_loc(start_loc)]
        self.alpha = alpha

    def select_move(self, move):
        # assert self.action_space.has_key(move), "move must be one of four reasonable movements"
        reward, observation = self.calculate_reward(self.action_space[move])
        return reward, observation

    def calculate_reward(self, move):
        tmp = self.data_base.geinitupian(self.time)
        dis = np.sqrt(np.square(self.start[0] - self.target[0]) + np.square(self.start[1] - self.target[1]))
        after = np.sqrt(np.square(self.start[0] + move[0] - self.target[0]) + np.square(self.start[1] + move[1] - self.target[1]))
        dis_reward = dis - after
        time_reward = - 2 * 50 / (tmp[self.start[0]+move[0], self.start[1]+move[1]]+1e-7) # 50 is block length
        reward = dis_reward * self.alpha + time_reward * (1 - self.alpha)
        self.time -= time_reward
        self.start = [self.start[0] + move[0], self.start[1] + move[1]]
        self.observation=[self.data_base.geinitupian(self.time), self.observation[1], self.target_and_loc(self.start)]
        return reward, self.observation

    def target_and_loc(self, loc, width = 5, block_size=[100, 100]):
        assert type(loc) == list and loc.__len__()==2 and loc[0]>=0 and loc[0]<=199 and loc[1]>=0 and loc[1]<=199, "Input must be a list, and it must has two eligible int number"
        tmp = np.zeros([block_size[0]+width*2, block_size[1]+width*2])
        cen1 = loc[0]+5
        cen2 = loc[1]+5
        tmp[cen1, cen2] = 60
        for i in range(1, 6):
            for j in range(0, i+1):
                tmp[cen1 + j, cen2 + (i-j)] = 60 - i * 10
                tmp[cen1 - j, cen2 + (i-j)] = 60 - i * 10
                tmp[cen1 + j, cen2 - (i-j)] = 60 - i * 10
                tmp[cen1 - j, cen2 - (i-j)] = 60 - i * 10
        return tmp[width:block_size[0]+width, width:block_size[1]+width]

def caijian(heng, zong):
    for i in range(20161101,20161131):
        name = str(i) + '.npy'
        tmp = np.load(name)
        saved = tmp[:, heng[0]:heng[1], zong[0]:zong[1]]
        np.save(str(i)+'_jietu', saved)

# caijian([99,199], [99,199])
tmp_env = env([21, 14], [45, 87], 999)
check = tmp_env.observation
a,b = tmp_env.select_move('shang')
print('test_finished')