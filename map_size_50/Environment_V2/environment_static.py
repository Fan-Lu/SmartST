import tkinter as tk
import os
import numpy as np
from PIL import Image, ImageTk

global punishment
punishment = -1
Reward = 1

class enviroment():
    """
    describtion: static environment
    dir_name: to indicate the dictionary of all numpy data files. (example: dir_name = "Environment")
    """

    def __init__(self, dir_name):
        self.load_files(dir_name)

    def load_files(self, dir_name):
        # initial function for load all data files and load mask data files.
        name = os.path.join(os.getcwd(), dir_name)
        a = os.listdir(name)
        files = [os.path.join(name, names) for names in a]
        for file_path in files:
            if os.path.isfile(file_path) and file_path.endswith('static_50.npy'):
                tmp = np.load(file_path)
                setattr(self, "database", tmp)
                value_point = []
                for i in range(tmp.shape[0]):
                    for j in range(tmp.shape[1]):
                        if tmp[i,j]!=0:
                            value_point.append([i, j])
            # if file_path.endswith('maskV1.npy'): # load mask file.
            #     tmp1 = np.load(file_path)
            #     value_point = []
            #     for i in range(tmp1.shape[0]):
            #         for j in range(tmp1.shape[1]):
            #             if tmp1[i,j]!=0:
            #                 value_point.append([i,j])
                setattr(self, "value_point", value_point) # value_point is all value point location

    def geinitupian(self, time):
        # interaction function for environment. Input is time and it is seconds
        return self.database

global enviroment_example  # a shared global load class for all env
enviroment_example = enviroment('Environment_V2')


class env:

    '''
    action space indicate all possible action
    initial parameters:
    start_loc: car initialization location , a list of two number within [0,99]
    target: car target location , a list of two number within [0,99]
    time: cars movement initial time, is int second.
    alpha: modify the balance between distance reward and negative time cost reward
    time_factor: modify the influence of negative time cost reward
    plot: if set true, our env will show a display windows.
    sleep: TO DO ISSUE
    '''
    def __init__(self, start_loc, target, time, alpha=0.9, time_factor=0.1, plot=True, sleep=0.5):
        self.start = start_loc
        self.record = start_loc
        self.target = target
        self.time = time
        self.data_base = enviroment_example
        self.observation = [self.data_base.geinitupian(time), self.target_and_loc(target), self.target_and_loc(start_loc)]
        self.alpha = alpha
        self.action_space_name = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
        self.action_space = {'up': [0, 1], 'upright': [1, 1], 'right': [1, 0], 'rightdown': [1, -1], 'down': [0, -1],
                    'downleft': [-1, -1], 'left': [-1, 0], 'leftup': [-1, 1]}
        if self.end_my_travel(start_loc):
            self.terminate = True
        else:
            self.terminate = False
        self.time_factor = time_factor
        self.plot = plot
        if self.plot:
            self.sleep = sleep
            self.cache = self.observation[0]
            self.root = tk.Tk()
            self.canvas = tk.Canvas(self.root, bg="white", height=900, width=900)
            self.canvas.pack()
            tmp = self.observation[0]
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            # tmp1.show()
            img = ImageTk.PhotoImage(tmp1)
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()

    def reset(self, start_loc, target, time):
        # To reset or env's location, target and time.
        self.start = start_loc
        self.record = start_loc
        self.target = target
        self.time = time
        self.observation = [self.data_base.geinitupian(time), self.target_and_loc(target), self.target_and_loc(start_loc)]
        if self.end_my_travel(start_loc):
            self.terminate = True
        else:
            self.terminate = False

        if self.plot:
            self.canvas.delete(self.canv_img)
            tmp = self.observation[0] - self.cache
            self.cache = self.observation[0]
            tmp = self.observation[0] * 2 +tmp * 20
            tmp[tmp < 0] = 0
            tmp[tmp >= 255] = 255
            tmp[self.start[0], self.start[1]] = 255
            tmp[self.target[0], self.target[1]] = 255
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            # tmp1.show()
            img = ImageTk.PhotoImage(tmp1)
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()
        return self.observation, self.get_moveable_list()

    def end_my_travel(self, loc):
        """
        Decide whether our interaction is done!
        :param loc: list
        :return: boolean
        """
        if self.observation[0][loc[0], loc[1]] == 0 or \
                np.sqrt(np.square(loc[0] - self.target[0])+np.square(loc[1] - self.target[1])) < 2:  # todo, change 3
            return True
        else:
            return False

    def step(self, move):
        # One step
        reward = self.calculate_reward(self.action_space[move])
        success = False

        if self.plot:
            self.canvas.delete(self.canv_img)
            tmp = self.observation[0] * 2
            tmp[self.start[0], self.start[1]] = 255
            tmp[self.target[0], self.target[1]] = 255
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            # tmp1.show()
            img = ImageTk.PhotoImage(tmp1)
            # img = ImageTk.PhotoImage(Image.fromarray(tmp).resize((800, 800)))
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()

        if self.end_my_travel(self.start) and self.observation[0][self.start[0], self.start[1]] != 0:
            reward = Reward
            self.terminate = True
            success = True

        return self.observation, reward, self.terminate, [self.start, self.target, self.get_moveable_list()], success

    def final_reward(self):
        finished_part = ((self.record[0] - self.start[0])**2 + (self.record[1] - self.start[1])**2)**0.5
        unfinished_part = ((self.target[0] - self.start[0])**2 + (self.target[1] - self.start[1])**2)**0.5
        reward = Reward * (unfinished_part/(unfinished_part + finished_part ))
        return reward

    def calculate_reward(self, move):
        #  todo, modify
        tmp = self.data_base.geinitupian(self.time)

        # todo
        dis = np.sqrt(np.square(self.start[0] - self.target[0]) + np.square(self.start[1] - self.target[1]))
        after = np.sqrt(np.square(self.start[0] + move[0] - self.target[0]) + np.square(self.start[1] + move[1] - self.target[1]))
        dis_reward = dis - after  # distance difference

        txx = tmp[self.start[0]+move[0], self.start[1]+move[1]]
        time_cost_xx = 50.0 / (txx + 0.0001)  # todo ?

        time_reward = - 50 / (tmp[self.start[0]+move[0], self.start[1]+move[1]] + 1e-7)  # 50 is block length
        if time_reward < -1e5:  # todo ?
            self.terminate = True
            self.observation = [self.observation[0], self.observation[1], self.target_and_loc(self.start)]
            final = self.final_reward()
            return punishment + final
        reward = dis_reward * self.alpha + self.time_factor * time_reward * (1 - self.alpha)
        self.time -= time_reward
        self.start = [self.start[0] + move[0], self.start[1] + move[1]]
        self.observation = [self.data_base.geinitupian(self.time), self.observation[1], self.target_and_loc(self.start)]
        return reward

    def get_moveable_list(self):
        """
        This function will return moveable list according to current state. e.g. filtering zero points
        :return: list
        """
        current_loc = [self.start[0], self.start[1]]
        current_velocity = self.observation[0]  # get velocity image

        '''
            no need for normalization here, because I normalization that numbers just for plot and UI.
            If you just use it to train, no need for this
        current_velocity[current_velocity < 0] = 0
        current_velocity[current_velocity >= 255] = 255
        '''
        r_l = [0 for _ in range(8)]
        # action_space = {'up': [0, 1], 'upright': [1, 1], 'right': [1, 0], 'rightdown': [1, -1], 'down': [0, -1],
        #             'downleft': [-1, -1], 'left': [-1, 0], 'leftup': [-1, 1]}
        i = 0
        for ele in self.action_space_name:
            temp_loc = np.array(current_loc) + np.array(self.action_space[ele])
            if temp_loc[0]<50 and temp_loc[0]>=0 and temp_loc[1]<50 and temp_loc[1]>=0 and current_velocity[temp_loc[0], temp_loc[1]] != 0:
                ## if we will enlarge the map later, we have to modify the 100 here
                r_l[i] = 1
            i += 1
        return np.array(r_l, dtype='float32')

    def target_and_loc(self, loc, width=5, block_size=[50, 50]):
        # translate our location into a matrix.
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
