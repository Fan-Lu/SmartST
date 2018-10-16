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
        elif type(games) == list and len(games) == len(maps):
            self.games = games
        else:
            raise NameError("parameter games must be a int or a list")

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
                        value_point.append([i, j])

            data_pool = []
            number_of_games = 0
            if type(games) == int:
                number_of_games = games
                for i in range(games):
                    random_index = np.random.randint(0, len(value_point)-1, 2)
                    if np.sum(np.abs(np.array(value_point[random_index[0]]) - np.array(value_point[random_index[1]]))) > 5:
                        data_pool.append([value_point[random_index[0]], value_point[random_index[1]]])
                    else:
                        i -= 1
            elif type(games) == dict:
                number_of_games = len(games[name])
                try:
                    for [index_0, index_1] in games[name]:
                        assert type(index_0) == int and type(index_1) == int
                        data_pool.append([index_0, index_1])
                except:
                    raise NameError("games list has a wrong format, each element in games list must be a two element list")
            self.maps.append([tmp, data_pool, np.zeros([number_of_games, 2])])

        self.only_when_success = only_when_success
        if only_when_success:
            self.success = False
        self.digital = digital
        assert reward_type == "one" or reward_type == "two", "wrong 'reward type' input, must be 'one' or 'two'. "
        self.reward_type = reward_type
        self.way_back = way_back
        self.running_reward = running_reward
        self.running_reward_interval = running_reward_interval
        self.plot = False
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, bg="white", height=900, width=900)
        self.canvas.pack()

        self.map_index = None  # current map of game
        self.game_index = None  # current game of a map
        self.last_loc = None  # last time location in current game
        self.current_loc = None  # current location in current game
        self.action_space_name = ['up', 'upright', 'right', 'rightdown', 'down', 'downleft', 'left', 'leftup']
        self.action_space = {'up': [0, 1], 'upright': [1, 1], 'right': [1, 0], 'rightdown': [1, -1], 'down': [0, -1],
                             'downleft': [-1, -1], 'left': [-1, 0], 'leftup': [-1, 1]}


    def figure_location(self, width = 5):
        tmp = np.zeros([self.map_size[0]+width*2, self.map_size[1]+width*2])
        cen1 = self.current_loc[0] + width
        cen2 = self.current_loc[1] + width
        tmp[cen1, cen2] = width * 10 + 10
        for i in range(1, 6):
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
                tmp[start] = 255
                tmp[target] = 255
                img = Image.fromarray(tmp).resize((800, 800))
                img.save("Map_{}_start:{}_target:{}.jpg".format(i, start, target), "jpeg")

    def final_reward(self):
        if self.reward_type == "one":

            return
        elif self.reward_type == "two":

            return

    def time_punishment(self, location):


    def moveable_list(self):
        tmp = self.maps[self.map_index]
        r_l = [0 for _ in range(8)]
        i = 0
        for ele in self.action_space_name:
            temp_loc = np.array(self.current_loc) + np.array(self.action_space[ele])
            if temp_loc[0]<self.map_size-1 and temp_loc[0]>=0 and temp_loc[1]<self.map_size-1 and temp_loc[1]>=0 \
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

        self.last_loc = None
        if map_number and game_number:
            self.current_loc =

        else:
            if self.only_when_success and self.success:
                map_rand_index = np.random.randint(0, len(self.maps)-1)
                game_rand_index = np.random.randint(0, len(self.maps[1])-1)

        self.plot = plot
        if self.plot:
            try:
                self.canvas.delete(self.canv_img)
            except:
                pass
            tmp = []
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            img = ImageTk.PhotoImage(tmp1)
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()
        else:
            try:
                self.canvas.delete(self.canv_img)
                self.root.update()
            except:
                pass


        if self.digital:
            return []
        else:
            return []



    def step(self):

        if self.plot:
            self.canvas.delete(self.canv_img)
            tmp = []
            tmp1 = Image.fromarray(tmp).resize((800, 800))
            img = ImageTk.PhotoImage(tmp1)
            self.canv_img = self.canvas.create_image(20, 20, anchor='nw', image=img)
            self.root.update()


        if self.digital:
            return []
        else:
            return []

















