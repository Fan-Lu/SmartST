
import numpy as np

class DataUnit(object):
    """
    This class to wrap data and label together
    """
    def __init__(self, data_cur, data_mid, data_hist, label):
        self.currnet = data_cur
        self.mid = data_mid
        self.hist = data_hist
        self.label = label

def data_loader(data,interval):
    """
    This function will transform time-based data in each data
    :param data: numpy [time1*100*100*2]
    :para interval:list[int],[12h,6h,0.05h] in paper,[1week,1day]
    :return: list[object] , DataUnit
    """

    time = np.shape(data)[0]

    assert time > 60*24*2, "Not 1 day data!!!"

    stride = 1  # sample rate
    total_num = int(time/2) # total size
    whole_data = []
    num_channel = int(interval[2]*60*2)

    for i in range(total_num+1, time, stride):
        label = data[i]

        # get current data
        t_data_cur = int(time/2) - num_channel
        data_cur = data[t_data_cur]
        for i in range(t_data_cur+1,t_data_cur + num_channel-1):
            data_cur = np.concatenate((data_cur,data[i+1]),axis=2)

        # get middle data
        t_data_mid = int(time/2) - int(interval[1]*60*2)
        data_mid = data[t_data_mid]
        for i in range(t_data_mid+1, t_data_mid + num_channel-1):
            data_mid = np.concatenate((data_mid, data[i+1]), axis=2)

        # get historical data
        t_data_hist = int(time/2) - int(interval[0]*60*2)
        data_hist = data[t_data_hist]
        for i in range(t_data_hist+1, t_data_hist + num_channel-1):
            data_hist = np.concatenate((data_hist, data[i+1]), axis=2)

        unit = DataUnit(data_cur,data_mid,data_hist,label)
        whole_data.append(unit)

    # TODO need to test whether this function is right

    return whole_data


def create_train_test(data):
    """
    This function will seperate data into training and test datset
    :param data:
    :return:
    """



if __name__ == '__main__':

    data = [[[[1, 1], [1, 1], [1, 1]],
             [[1, 1], [1, 1], [1, 1]],
             [[1, 1], [1, 1], [1, 1]]],

            [[[2, 2], [2, 2], [2, 2]],
             [[2, 2], [2, 2], [2, 2]],
             [[2, 2], [2, 2], [2, 2]]],

            [[[3, 3], [3, 3], [3, 3]],
             [[3, 3], [3, 3], [3, 3]],
             [[3, 3], [3, 3], [3, 3]]],

            [[[41, 41], [4, 4], [4, 4]],
             [[42, 42], [4, 4], [4, 4]],
             [[43, 43], [4, 4], [4, 4]]],

            [[[5, 5], [5, 5], [5, 5]],
             [[5, 5], [5, 5], [5, 5]],
             [[5, 5], [5, 5], [5, 5]]]
            ] # 5*3*3*2



    intervals = [1,1,3]

    data1 = data[0]
    data2 = data[1]
    data3 = data[2]



    for i in range(2,np.shape(data)[0]-1):
        data3 = np.concatenate((data3,data[i+1]),axis=2)
    print (data3)
