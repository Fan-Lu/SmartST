import os
import numpy as np
import time

file_dir = "di_data"

'''
due to the reasons of storage. All file in processed separately.  This will cause all the data lose
some records information at beginning of one day. That will depends on the data record methods.
'''

def preprocess(file_dir):
    dir = os.path.join(os.getcwd(), file_dir)
    file_list = get_files(dir)
    wishdata(file_list)
    return 0


def get_files(dir):
    result = []
    data_files = [os.path.join(dir, name) for name in os.listdir(dir)]
    for files in data_files:
        if os.path.isfile(files) and files.endswith(".txt"):
            result.append(files)
    return result


def datetime(dt):
    s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    return int(s)


def wishdata(file_list):
    for file in file_list:
        print(file[-12:-4])
        tmp_time_begin = file[-12:-8] + '-' + file[-8:-6] + '-' + file[-6:-4] + ' 00:00:00'
        tmp_time_end = file[-12:-8] + '-' + file[-8:-6] + '-' + file[-6:-4] + ' 23:59:59'
        begin_time = datetime(tmp_time_begin) - 43200 # dong ling shi -46800
        end_time = datetime(tmp_time_end) - 43200 # xia ling shi -43200
        print("begin time: ", begin_time)
        print("end time: ", end_time)
        rator = Generator(min_time=begin_time, max_time=end_time, min_longitude=104.04210, min_latitude=30.65282,
                          interval=50, time_interval=600)
        print(rator.data_array.shape)
        rator.main_one_file(file)


class Generator:
    def __init__(self, min_time, max_time, min_longitude, min_latitude, interval, time_interval, block_length=[200, 200]):
        # min_time and max_time will record the raw data file's time interval,
        # it should be unix time record in the file.
        # for this task, min_longitude and min_latitude will be the same for all raw data file.
        # We set this two value as min_longitude = 3408, min_latitude = 11568.9
        # interval is location block size, we set 50, means 50 meters. time interval is 30, means 30 seconds.
        self.min_time = min_time
        self.max_time = max_time
        self.min_longitude = min_longitude
        self.min_latitude = min_latitude
        self.interval = interval
        self.time_interval = time_interval
        self.block = block_length
        self.counter = 0
        self.calcu = int((self.max_time - self.min_time)/self.time_interval)+1
        self.data_array = np.zeros([self.calcu, *self.block, 2])  # 0 is inflow, 1 is outflow

    def transfer(self, date, longitude, latitude):
        date = int((date - self.min_time) / self.time_interval)
        longitude = int((longitude - self.min_longitude) * 3.1415 / 180 * 6371 / self.interval * 1000)
        latitude = int(((latitude - self.min_latitude) * 3.1415 / 180 * 6371) / self.interval * 1000)
        return date, longitude, latitude

    def process_line(self, line):
        try:
            Car_ID, order_ID, Date, longitude, latitude = line.split(',')
            Date, longitude, latitude = self.transfer(float(Date), float(longitude), float(latitude))
            return [Car_ID, order_ID, Date, longitude, latitude]
        except ValueError:
            return [0, 0, 0, 0, 0]

    def wrong(self, context):
        if (context[2]>=0 and context[2]<self.calcu) and (context[3]>=0 and context[3]<self.block[0]) and (context[4]>=0 and context[4]<self.block[1]):
            return False
        else:
            return True

    def matrix(self, tmp):
        self.counter += 1
        for i in range(len(tmp)-1):  # calculate outflow
            if tmp[i][3] != tmp[i+1][3] or tmp[i][4] != tmp[i+1][4]:
                tmp_tmp = self.data_array[tmp[i][2], tmp[i][3], tmp[i][4], 1]
                self.data_array[tmp[i][2], tmp[i][3], tmp[i][4], 1] += 1
                tmp_tmp = self.data_array[tmp[i][2], tmp[i][3], tmp[i][4], 1]
        for i in range(1, len(tmp)):  # calculate inflow
            if tmp[i][3] != tmp[i-1][3] or tmp[i][4] != tmp[i-1][4]:
                tmp_tmp = self.data_array[tmp[i][2], tmp[i][3], tmp[i][4], 0]
                self.data_array[tmp[i][2], tmp[i][3], tmp[i][4], 0] += 1
                tmp_tmp = self.data_array[tmp[i][2], tmp[i][3], tmp[i][4], 0]

    def main_one_file(self, file):
        file_context = open(file)
        tmp = []
        line = file_context.readline()
        while line:
            context = self.process_line(line)
            if context[0] == 0 or self.wrong(context):
                pass
            elif tmp == [] or tmp[-1][1] == context[1]:
                tmp.append(context)
            elif tmp[-1][1] != context[1]:
                self.matrix(tmp)
                tmp = [context]
            line = file_context.readline()
        print('The number of ', file[-12:-4], ' trajectory we have calculated is: ', self.counter)
        np.save(file[-12:-4], self.data_array)

preprocess(file_dir)






