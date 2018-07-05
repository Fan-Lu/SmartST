
import numpy as np
import os
import Config as config


class DataUnit(object):
    """
    This class to wrap data, label together, extra factors
    """
    def __init__(self, data_close, data_period, label, week:str):
        self.close = data_close
        self.period = data_period
        self.label = label
        self.week = week

        # extra factor

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

    for i in range(0, total_num, stride):
        # get label
        label = data[i+interval[2]]

        # get closeness data
        t_data_close = interval[1]
        data_close = data[i+t_data_close]
        for j in range(i+t_data_close+1, i+t_data_close+num_channel):
            data_close = np.concatenate((data_close,data[j]),axis=2)

        # get period data
        data_per = data[i]
        for j in range(i+1, i+num_channel):
            data_per = np.concatenate((data_per, data[j]), axis=2)

        unit = DataUnit(data_close, data_per, label, str(get_week(i+interval[2])))
        whole_data.append(unit)

    return whole_data


def create_train_test(data):
    """
    This function will seperate data into training and test datset
    :param data:
    :return:
    """
    # todo

def get_week(n):
    """
    This funciton will output week information
    :return: int
    """
    base_date = 2 # Tuesday

    return base_date+int(n // config.time)%7














