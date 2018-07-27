import os
import numpy as np
import time

file_dir = "di_data(Fake)"
bushi = 46800


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
        begin_time = datetime(tmp_time_begin) - bushi # dong ling shi -46800
        end_time = datetime(tmp_time_end) - bushi # xia ling shi -43200
        print("begin time: ", begin_time)
        print("end time: ", end_time)
        rator = Generator(min_time=begin_time, max_time=end_time, min_longitude=104.04210, min_latitude=30.65282,
                          interval=50, time_interval=300)
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
        times = int((date - self.min_time) / self.time_interval)
        long = int((longitude - self.min_longitude) * 3.1415 / 180 * 6371 / self.interval * 1000)
        la = int(((latitude - self.min_latitude) * 3.1415 / 180 * 6371) / self.interval * 1000)
        longitude = (longitude - self.min_longitude) * 3.1415 / 180 * 6371 * 1000
        latitude = ((latitude - self.min_latitude) * 3.1415 / 180 * 6371) * 1000
        return long, la, times, longitude, latitude

    def process_line(self, line):
        try:
            Car_ID, order_ID, Date, longitude_tmp, latitude_tmp = line.split(',')
            long, la, times, longitude, latitude = self.transfer(float(Date), float(longitude_tmp), float(latitude_tmp))
            return [Car_ID, order_ID, times, long, la, Date, longitude, latitude]
        except ValueError:
            return [0, 0, 0, 0, 0, 0, 0, 0]

    def wrong(self, context):
        if (context[2]>=0 and context[2]<self.calcu) and (context[3]>=0 and context[3]<self.block[0]) and (context[4]>=0 and context[4]<self.block[1]):
            return False
        else:
            return True

    def speed(self, t1, x1, y1, t2, x2, y2):
        t1 = int(t1)
        t2 = int(t2)
        sudu = np.sqrt((x1-x2)**2 + (y1-y2)**2) / (t2 - t1)
        return sudu

    def matrix(self, tmp):
        self.counter += 1
        for i in range(1,len(tmp)):
            sudu = self.speed(tmp[i-1][5], tmp[i-1][6], tmp[i-1][7], tmp[i][5], tmp[i][6], tmp[i][7])
            self.data_array[tmp[i][2], tmp[i][3], tmp[i][4], 0] += sudu
            self.data_array[tmp[i][2], tmp[i][3], tmp[i][4], 1] += 1

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
        self.data_array[:, :, :, 1] += 1e-7
        self.data_array[:, :, :, 0] /= self.data_array[:, :, :, 1]
        np.save(file[-12:-4], self.data_array[:, :, :, 0])

preprocess(file_dir)
