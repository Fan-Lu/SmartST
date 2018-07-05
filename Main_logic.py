
import os
import Config as config
import numpy as np
from Utils import data_loader


if __name__ == '__main__':

    #### load original data
    pa_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) # get last state file name
    name1 = '/data/data(normalized)/'
    name_start = 20161101
    num_file = 30
    path1 = pa_path + name1 + str(name_start)+'(normalized).npy'
    temp_data = np.load(path1)
    print("loading " + str(name_start) + "th file!")
    for i in range(name_start+1, name_start+num_file):
        print("loading "+str(i)+"th file!")
        path = pa_path + name1 + str(i)+'(normalized).npy'
        temp_data = np.concatenate((temp_data, np.load(path)), axis=0)
    print("number of data is: "+str(np.shape(temp_data)[0]))
    assert np.shape(temp_data)[0] == num_file*config.time, "Not reasonable data!!!"

    intervals = config.intervals # [1 day,20 min,143(label)]

    #### apply data_loader to pre-process data that fit to our network
    # use for test
    processed_data = data_loader(temp_data,intervals) # (4182,),c:(200, 200, 4),p:(200, 200, 4),l:(200, 200, 2)

    #### could be deleted
    # print(np.shape(processed_data[0].close),np.shape(processed_data[0].period),np.shape(processed_data[0].label))
    # print(len(processed_data))
    # print(processed_data[0].week)
    # for j in range(30):
    #     print('---------------------------')
    #     if j>0:
    #         print(processed_data[144*j-1].week)
    #     print(processed_data[144*j].week)
    #     print(processed_data[144*j+1].week)
