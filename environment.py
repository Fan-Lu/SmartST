import os
import numpy as np
import csv

global punishment
punishment = -1000


class enviroment(): 
    def __init__(self, dir_name):
        self.load_files(dir_name)

    def load_files(self, dir_name):
        name = os.path.join(os.getcwd(), dir_name)
        a = os.listdir(name)
        files = [os.path.join(name, names) for names in a]
        for file_path in files:
            if os.path.isfile(file_path) and file_path.endswith('.npy') and not file_path.endswith('mask.npy'):
                tmp = np.load(file_path)
                setattr(self, file_path[-12:-4], tmp)
            if file_path.endswith('maskV1.npy'):
                tmp1 = np.load(file_path)
                value_point = []
                for i in range(tmp1.shape[0]):
                    for j in range(tmp1.shape[1]):
                        if tmp1[i,j]!=0:
                            value_point.append([i,j])
                setattr(self, "value_point", value_point)

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
            writer.writerow([data_array[0]])

global enviroment_example
enviroment_example = enviroment('Environment_V2')

class env:
    action_space = {'shang': [0, 1], 'you': [1, 0], 'xia': [0, -1], 'zuo': [-1, 0]}
    def __init__(self, start_loc, target, time, alpha = 0.5, time_factor= 0.1):
        self.start = start_loc
        self.target = target
        self.time = time
        self.data_base = enviroment_example
        self.observation = [self.data_base.geinitupian(time), self.target_and_loc(target), self.target_and_loc(start_loc)]
        self.alpha = alpha
        if self.end_my_travel(start_loc):
            self.terminate = True
        else:
            self.terminate = False
        self.time_factor = time_factor

    def reset(self, start_loc, target, time):
        self.start = start_loc
        self.target = target
        self.time = time
        self.observation = [self.data_base.geinitupian(time), self.target_and_loc(target), self.target_and_loc(start_loc)]
        if self.end_my_travel(start_loc):
            self.terminate = True
        else:
            self.terminate = False

    def end_my_travel(self, loc):
        if self.observation[0][loc[0], loc[1]] == 0 or np.sqrt(np.square(loc[0] - self.target[0])+np.square(loc[1] - self.target[1]))<3 :
            a = self.observation[0][loc[0], loc[1]]
            b = np.sqrt(np.square(loc[0] - self.target[0])+np.square(loc[1] - self.target[1]))
            return True
        else:
            return False

    def select_move(self, move):
        # assert self.action_space.has_key(move), "move must be one of four reasonable movements"
        reward = self.calculate_reward(self.action_space[move])
        return reward, self.observation, self.terminate

    def calculate_reward(self, move):
        tmp = self.data_base.geinitupian(self.time)
        dis = np.sqrt(np.square(self.start[0] - self.target[0]) + np.square(self.start[1] - self.target[1]))
        after = np.sqrt(np.square(self.start[0] + move[0] - self.target[0]) + np.square(self.start[1] + move[1] - self.target[1]))
        dis_reward = dis - after

        txx = tmp[self.start[0]+move[0], self.start[1]+move[1]]
        time_cost_xx = 50 / txx


        time_reward = - 50 / (tmp[self.start[0]+move[0], self.start[1]+move[1]] + 1e-7 ) # 50 is block length
        if time_reward < punishment:
            self.terminate = True
            self.observation = [self.observation[0], self.observation[1], self.target_and_loc(self.start)]
            return punishment
        reward = dis_reward * self.alpha + self.time_factor * time_reward * (1 - self.alpha)
        self.time -= time_reward
        self.start = [self.start[0] + move[0], self.start[1] + move[1]]
        self.observation=[self.data_base.geinitupian(self.time), self.observation[1], self.target_and_loc(self.start)]
        return reward

    def target_and_loc(self, loc, width = 5, block_size=[100, 100]):
        assert type(loc) == list and loc.__len__()==2 and loc[0]>=0 and loc[0]<=(block_size[0]-1) and loc[1]>=0 and loc[1]<=(block_size[1]-1), "Input must be a list, and it must has two eligible int number"
        tmp = np.zeros([block_size[0]+width*2, block_size[1]+width*2])
        cen1 = loc[0]+width
        cen2 = loc[1]+width
        tmp[cen1, cen2] = width * 10 + 10
        for i in range(1, 6):
            for j in range(0, i+1):
                tmp[cen1 + j, cen2 + (i-j)] = (width - i) * 10 + 10
                tmp[cen1 - j, cen2 + (i-j)] = (width - i) * 10 + 10
                tmp[cen1 + j, cen2 - (i-j)] = (width - i) * 10 + 10
                tmp[cen1 - j, cen2 - (i-j)] = (width - i) * 10 + 10
        return tmp[width:block_size[0]+width, width:block_size[1]+width]

def caijian(heng, zong):
    for i in range(20161101,20161131):
        name = str(i) + '.npy'
        tmp = np.load(name)
        saved = tmp[:, heng[0]:heng[1], zong[0]:zong[1]]
        np.save(str(i)+'_jietu', saved)

# caijian([99,199], [99,199])
tmp_env = env([28, 21], enviroment_example.value_point[666], 999)
enviroment_example.value_point[999]
check = tmp_env.observation
a, b, c = tmp_env.select_move('shang')
print('test_finished')