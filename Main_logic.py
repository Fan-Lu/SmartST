
import os
import Config as config
import numpy as np
from Utils import data_loader


if __name__ == '__main__':

    #### load original data
    pa_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) # get last state file name
    name1 = '/data/'
    name_start = 20161101
    num_file = 3
    path1 = pa_path + name1 + str(name_start)+'.npy'
    temp_data = np.load(path1)
    for i in range(name_start+1, name_start+num_file):
        path = pa_path + name1 + str(i)+'.npy'
        temp_data = np.concatenate((temp_data, np.load(path)), axis=0)

    assert np.shape(temp_data)[0] == num_file*config.time, "Not reasonable data!!!"

    intervals = config.intervals # [1 day,20 min,143(label)]

    #### apply data_loader to pre-process data that fit to our network
    # use for test
    processed_data = data_loader(temp_data,intervals)

    print(np.shape(processed_data[0].close),np.shape(processed_data[0].period),np.shape(processed_data[0].label))
    print(len(processed_data))
    print(processed_data[0].week)
    print(processed_data[143].week)
    print(processed_data[144].week)
    print(processed_data[286].week)
    print(processed_data[287].week)