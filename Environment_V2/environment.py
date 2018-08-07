import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import time
import tkinter as tk

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
    action_space = {'up': [0, 1], 'upright': [1, 1], 'right': [1, 0], 'rightdown': [1, -1], 'down': [0, -1], 'downleft':[-1, -1], 'left': [-1, 0], 'leftup': [-1, 1]}
    def __init__(self, start_loc, target, time, alpha = 0.5, time_factor= 0.1, plot = True, sleep = 0.5):
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
        if plot:
            self.sleep = sleep
            self.plot = plot
            self.root = tk.Tk()
            self.canvas = tk.Canvas(self.root, bg="white",height=900, width=900)
            self.canvas.pack()
            tmp = self.observation[0] * 3
            tmp[self.start[0], self.start[1]] = 255
            tmp[self.target[0], self.target[1]] = 255
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            # tmp1.show()
            img = ImageTk.PhotoImage(tmp1)
            # img = ImageTk.PhotoImage(Image.fromarray(tmp).resize((800, 800)))\
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()


    def reset(self, start_loc, target, time):
        self.start = start_loc
        self.target = target
        self.time = time
        self.observation = [self.data_base.geinitupian(time), self.target_and_loc(target), self.target_and_loc(start_loc)]
        if self.end_my_travel(start_loc):
            self.terminate = True
        else:
            self.terminate = False

        if self.plot:
            self.canvas.delete(self.canv_img)
            tmp = self.observation[0] * 3
            tmp[self.start[0], self.start[1]] = 255
            tmp[self.target[0], self.target[1]] = 255
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            # tmp1.show()
            img = ImageTk.PhotoImage(tmp1)
            # img = ImageTk.PhotoImage(Image.fromarray(tmp).resize((800, 800)))
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()
        return self.observation

    def end_my_travel(self, loc):
        if self.observation[0][loc[0], loc[1]] == 0 or np.sqrt(np.square(loc[0] - self.target[0])+np.square(loc[1] - self.target[1]))<3 :
            a = self.observation[0][loc[0], loc[1]]
            b = np.sqrt(np.square(loc[0] - self.target[0])+np.square(loc[1] - self.target[1]))
            return True
        else:
            return False


    def step(self, move):
        # assert self.action_space.has_key(move), "move must be one of four reasonable movements"
        reward = self.calculate_reward(self.action_space[move])

        if self.plot:
            self.canvas.delete(self.canv_img)
            tmp = self.observation[0] * 3
            tmp[self.start[0], self.start[1]] = 255
            tmp[self.target[0], self.target[1]] = 255
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            # tmp1.show()
            img = ImageTk.PhotoImage(tmp1)
            # img = ImageTk.PhotoImage(Image.fromarray(tmp).resize((800, 800)))
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()

        if self.end_my_travel(self.start):
            reward = 1000
            self.terminate = True

        return self.observation, reward, self.terminate, [self.start, self.target]

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

if __name__ == '__main__':
    # caijian([99,199], [99,199])
    tmp_env = env([28, 21], enviroment_example.value_point[666], 999)
    enviroment_example.value_point[999]
    check = tmp_env.observation
    a, b, c = tmp_env.step('up')
    print('test_finished')