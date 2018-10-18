import tkinter as tk
import os
import numpy as np
from PIL import Image, ImageTk


class environment:
    def __init__(self, maps, map_size, games, only_when_success, digital, reward_type, way_back, running_reward, running_reward_interval):
        """
        maps: list contains string, which is the name of map file
        map_size: specific the shape of array.
        games: a dict contain list or int type, we can specific different start location and target in each map through
                this command. If it is a list of lists, we will play the game with the location information in each list.
                If it is an int type number, we will randomly select the N random location.
        only_when_success: (bool type) if true, we will only switch to next game when we win this try
        digital: (bool type) if True, the representation of location information is digital. Else, is figure
        reward_type: ("one" or "two") order one reward function or order two reward function
        way_back: (bool type) if True, our agent is allowed to back to the location in last step
        running_reward: (bool type) if True, A running reward baseline is set for each task in each map
        running_reward_interval: (int type) how many episode passed when we update our interval value
        """
        if type(games) == int:
            self.games = games
        elif type(games) == dict and len(games) == len(maps):
            self.games = games
        else:
            raise NameError("parameter games must be a int or a dict")

        self.maps = []
        self.map_size = map_size
        for name in sorted(maps):
            name = os.path.join(os.getcwd(), name)
            try:
                tmp = np.load(name)
            except:
                raise ValueError("the file {} is not a standard map file".format(name))
            assert tmp.shape == map_size, "when we load {}, this map size not equals what we set".format(name)
            value_point = []
            for i in range(map_size[0]):
                for j in range(map_size[1]):
                    if tmp[i, j] != 0:
                        value_point.append(np.array([i, j]))

            data_pool = []
            number_of_games = 0
            if type(games) == int:
                number_of_games = games
                for i in range(games):
                    random_index = np.random.randint(0, len(value_point)-1, 2)
                    if np.sum(np.abs(value_point[random_index[0]] - value_point[random_index[1]])) > 5:
                        data_pool.append([value_point[random_index[0]], value_point[random_index[1]]])
                    else:
                        i -= 1
            elif type(games) == dict:
                if type(games[name]) == int:
                    number_of_games = games[name]
                    for i in range(number_of_games):
                        random_index = np.random.randint(0, len(value_point) - 1, 2)
                        if np.sum(np.abs(
                                np.array(value_point[random_index[0]]) - np.array(value_point[random_index[1]]))) > 5:
                            data_pool.append([value_point[random_index[0]], value_point[random_index[1]]])
                        else:
                            i -= 1
                elif type(games[name]) == list:
                    try:
                        number_of_games = len(games[name])
                        for i in games[name]:
                            assert type(i[0]) == int and type(i[1]) == int, "element in dict must be int"
                            data_pool.append([value_point[i[0]], value_point[i[1]]])
                    except:
                        raise NameError("games list has a wrong format, each element in games list must be a two element list")
            self.maps.append([tmp, data_pool, np.zeros([number_of_games, 3])])
            # this 3 first: current running_reward, second: accumulate running reward, third: times counted in running reward

        self.only_when_success = only_when_success
        if only_when_success:
            self.success_flag = True
        self.digital = digital
        assert reward_type == "one" or reward_type == "two", "wrong 'reward type' input, must be 'one' or 'two'. "
        self.reward_type = reward_type
        self.way_back = way_back
        self.running_reward = running_reward
        self.running_reward_interval = running_reward_interval
        self.plot = False
        # self.root = tk.Tk()
        # self.canvas = tk.Canvas(self.root, bg="white", height=900, width=900)
        # self.canvas.pack()

        self.map_index = None  # current map of game
        self.game_index = None  # current game of a map
        self.current_game_start = None  # the start location in current game
        self.current_game_target = None  # the target location in current game
        self.last_loc = None  # last time location in current game
        self.current_loc = None  # current location in current game
        self.action_space_name = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
        self.action_space = {'up': [-1, 0], 'upright': [-1, 1], 'right': [0, 1], 'rightdown': [1, 1], 'down': [1, 0],
                             'downleft': [1, -1], 'left': [0, -1], 'leftup': [-1, -1]}

        self.max_reward = 1
        self.time_factor = 1

    def figure_location(self, location, width = 5):
        tmp = np.zeros([self.map_size[0]+width*2, self.map_size[1]+width*2])
        cen1 = location[0] + width
        cen2 = location[1] + width
        tmp[cen1, cen2] = width * 10 + 10
        for i in range(1, 1+width):
            for j in range(0, i+1):
                tmp[cen1 + j, cen2 + (i - j)] = (width - i) * 10 + 10
                tmp[cen1 - j, cen2 + (i - j)] = (width - i) * 10 + 10
                tmp[cen1 + j, cen2 - (i - j)] = (width - i) * 10 + 10
                tmp[cen1 - j, cen2 - (i - j)] = (width - i) * 10 + 10
        return tmp[width:self.map_size[0] + width, width:self.map_size[1] + width]


    def print_games(self):
        for i in range(len(self.maps)):
            a_map = self.maps[i]
            tmp = a_map[0]
            for [start, target] in a_map[1]:
                tmp = int(tmp * 4)
                tmp[start[0], start[1]] = 255
                tmp[target[0], target[1]] = 255
                img = Image.fromarray(tmp).resize((800, 800))
                img.save("Map_{}_start:{}_target:{}.jpg".format(i, start, target), "jpeg")

    def final_reward(self):
        start = self.maps[self.map_index][1][self.game_index][0]
        target = self.maps[self.map_index][1][self.game_index][1]
        current = self.current_loc
        total_distance = np.sqrt(np.sum(np.square(start - target)))
        current_distance = np.sqrt(np.sum(np.square(current - target)))
        if self.reward_type == "one":
            R = (total_distance - current_distance) / total_distance * self.max_reward
        elif self.reward_type == "two":
            R = np.square((total_distance - current_distance) / total_distance * self.max_reward)
        if self.running_reward:
            self.maps[self.map_index][2][self.game_index, 1] += R
            self.maps[self.map_index][2][self.game_index, 2] += 1
            if self.maps[self.map_index][2][self.game_index, 2] >= self.running_reward_interval:
                self.maps[self.map_index][2][self.game_index, 0] = self.maps[self.map_index][2][self.game_index, 1] / self.maps[self.map_index][2][self.game_index, 2]
                self.maps[self.map_index][2][self.game_index, 1] = 0
                self.maps[self.map_index][2][self.game_index, 2] = 0
        return R

    def time_punishment(self, location):  # calculate the time punishment
        speed = self.maps[self.map_index][0][location[0], location[1]]
        return self.time_factor * 50.0 / speed

    def moveable_list(self):
        tmp = self.maps[self.map_index][0]
        r_l = [0 for _ in range(8)]
        i = 0
        if self.way_back and self.last_loc is not None:
            for ele in self.action_space_name:
                temp_loc = self.current_loc + np.array(self.action_space[ele])
                if temp_loc[0]!=self.last_loc[0] and temp_loc[1]!=self.last_loc[1] and temp_loc[0]<=self.map_size[0] \
                        and temp_loc[0]>=0 and temp_loc[1]<=self.map_size[1] and temp_loc[1]>=0 and tmp[temp_loc[0], temp_loc[1]] != 0:
                    r_l[i] = 1
                i += 1
        else:
            for ele in self.action_space_name:
                temp_loc = self.current_loc + np.array(self.action_space[ele])
                if temp_loc[0]<self.map_size[0] and temp_loc[0]>=0 and temp_loc[1]<self.map_size[1] and temp_loc[1]>=0 \
                and tmp[temp_loc[0], temp_loc[1]] != 0:
                    r_l[i] = 1
                i += 1
        return np.array(r_l, dtype='float32')

    def reset(self, map_number=None, game_number=None, plot=False):
        """
        map_number: specify the index of map we play
        game_number: specify the index of game in the specify map
        plot: if True, we will plot our map
        """

        if map_number and game_number:
            self.map_index = map_number
            self.game_index = game_number
        else:
            if self.only_when_success:
                if self.success_flag:
                    self.map_index = np.random.randint(0, len(self.maps))
                    self.game_index = np.random.randint(0, self.maps[self.map_index][2].shape[0])
                    self.success_flag = False
            else:
                self.map_index = np.random.randint(0, len(self.maps))
                self.game_index = np.random.randint(0, self.maps[self.map_index][2].shape[0])

        self.current_game_start = self.maps[self.map_index][1][self.game_index][0]
        self.current_game_target = self.maps[self.map_index][1][self.game_index][1]
        self.last_loc = None
        self.current_loc = self.maps[self.map_index][1][self.game_index][0]

        if self.plot and not plot:
            self.root.destroy()
        if not self.plot and plot:
            self.root = tk.Tk()
            self.canvas = tk.Canvas(self.root, bg="white", height=900, width=900)
            self.canvas.pack()

        if plot:
            tmp = self.maps[self.map_index][0] * 4
            tmp[self.current_game_start[0], self.current_game_start[1]] = 255
            tmp[self.current_game_target[0], self.current_game_target[1]] = 255
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            img = ImageTk.PhotoImage(tmp1)
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()
        self.plot = plot

        if self.digital:
            location_information = [self.current_loc, self.current_game_target]
        else:
            location_information = [self.figure_location(self.current_loc),
                                    self.figure_location(self.current_game_target)]

        return self.maps[self.map_index][0], location_information, self.moveable_list()



    def step(self, move, last_step = False, test = False):
        self.last_loc = self.current_loc
        self.current_loc += np.array(self.action_space[move])
        time_punish = self.time_punishment(self.current_loc)

        if self.plot:
            self.canvas.delete(self.canv_img)
            tmp = self.maps[self.map_index][0] * 4
            tmp[self.current_game_start[0], self.current_game_start[1]] = 255
            tmp[self.current_game_target[0], self.current_game_target[1]] = 255
            tmp[self.current_loc[0], self.current_loc[1]] = 255
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            img = ImageTk.PhotoImage(tmp1)
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()

        reach_target = (np.sum(np.abs(self.current_game_target - self.current_loc)) < 2)

        if reach_target:
            self.success_flag = True

        if last_step or reach_target and not test:
            final_R = self.final_reward()
            if self.running_reward:
                running_final_R = self.maps[self.map_index][2][self.game_index, 0]
            else:
                running_final_R = None
        else:
            final_R = None
            running_final_R = None

        if self.digital:
            location_information = [self.current_loc, self.current_game_target]
        else:
            location_information = [self.figure_location(self.current_loc), self.figure_location(self.maps[self.map_index][0][self.current_game_target])]

        return self.maps[self.map_index][0], location_information, self.moveable_list(), -time_punish, reach_target, final_R, running_final_R
















