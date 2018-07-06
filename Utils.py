
import numpy as np
import Config as config
import csv
import os

class DataUnit(object):
    """
    This class to wrap data, label together, extra factors
    """
    def __init__(self, data_close, data_period, label, tpr, week: str):
        self.close = data_close
        self.period = data_period
        self.label = label
        self.week = week # date in week
        self.tpr = tpr # temperature
        self.weather = None

    def oneHot_weather(self, type_list: list, target: str):

        weather_data = [0] * len(type_list)
        weather_data[type_list.index(target)] = 1.0
        self.weather = np.array(weather_data, dtype=float)



def data_loader(data, interval):
    """
    This function will transform time-based data in each data
    :param data: numpy [time1*100*100*2]
    :para interval:list[int],[+0,+141]
    :return: list[object] , DataUnit
    """

    time = np.shape(data)[0]

    stride = 1  # sample rate
    total_num = time - config.time # total size
    whole_data = []
    num_channel = 2

    temperature, condition, type_list = getTemperature() # (721,1),(721,1),(9,1)
    max_temp = max(temperature)

    for i in range(0, total_num, stride):
        # get label
        label = data[i+interval[2]]
        temp_each = temperature[i//6]/max_temp
        type_each = condition[i//6]

        # get closeness data
        t_data_close = interval[1]
        data_close = data[i+t_data_close]
        for j in range(i+t_data_close+1, i+t_data_close+num_channel):
            data_close = np.concatenate((data_close,data[j]),axis=2)

        # get period data
        data_per = data[i]
        for j in range(i+1, i+num_channel):
            data_per = np.concatenate((data_per, data[j]), axis=2)

        unit = DataUnit(data_close, data_per, label, temp_each, str(get_week(i+interval[2])))
        unit.oneHot_weather(type_list, type_each)
        whole_data.append(unit)

    return whole_data


def create_train_test(data):
    """
    This function will separate data into training(0.9) and test(0.1) data-set
    :param data: list[object]
    :return: list[object],list[object]
    """
    num_data = len(data)
    np.random.shuffle(data)
    train_data = data[0:int(0.9*num_data)]
    test_data = data[int(0.9*num_data):]
    return train_data, test_data

def get_week(n):
    """
    This funciton will output week information
    :return: int
    """
    base_date = 2 # Tuesday

    return (base_date+int(n//config.time))%7


def getTemperature():
    """
    This function is used for collect weather info.
    :return:  list[int],list[str],list[str]
    """
    path = "data/weather.csv"
    temperature, condition = [], []
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=' ', quotechar='|')
        for row in reader:
            temperature.append(int(row[0]))
            condition.append(row[1][3:].rstrip('.'))

    type_list = []
    for ele in condition:
        if ele not in type_list:
            type_list.append(ele)
    return temperature, condition, type_list

class weather(nn.Module):
    def __init__(self, input_vector_size):
        super(weather, self).__init__()
        self.input_vector = input_vector_size
        self.linear = nn.Linear(in_features=self.input_vector, out_features= 49)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=5)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=5, stride=2)
        self.deconv5 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5, stride=3)
        self.deconv6 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4)
        self.relu = nn.ReLU()

    def forward(self, weather_vector):
        weather = self.relu(self.linear(weather_vector)).reshape(-1, 1, 7, 7)
        weather = self.relu(self.deconv1(weather))
        weather = self.relu(self.deconv2(weather))
        weather = self.relu(self.deconv3(weather))
        weather = self.relu(self.deconv4(weather))
        weather = self.relu(self.deconv5(weather))
        weather = self.deconv6(weather)

        return weather












