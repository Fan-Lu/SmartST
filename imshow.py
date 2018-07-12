import os
import Config as config
import numpy as np
from Utils import *
from model_ww import STResNet
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description='SmartST')
parser.add_argument('--model-dir', metavar='DIR', help='path to data', default='/mnt/data//fan/SmartST/model_saved_ww/')
parser.add_argument('--result-dir', metavar='DIR', help='path to data', default='/mnt/data//fan/SmartST/result_saved_ww/')
# parser.add_argument('--model-dir', metavar='DIR', help='path to data', default='/home/exx/Lab/SmartST/model_saved/')
# parser.add_argument('--result-dir', metavar='DIR', help='path to data', default='/home/exx/Lab/SmartST/result_saved/')
parser.add_argument('--use-plt', default=False, type=bool, help='plot figure')
parser.add_argument('--batch-size', default=16, type=int, help='batch size default=32')
args = parser.parse_args()

#### load original data
pa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))  # get last state file name
name1 = '/SmartST/data/data(normalized)/'
name_start = 20161101
num_file = 3
path1 = pa_path + name1 + str(name_start) + '(normalized).npy'
temp_data = np.load(path1)
print("loading " + str(name_start) + "th file!")
for i in range(name_start + 1, name_start + num_file):
    print("loading " + str(i) + "th file!")
    path = pa_path + name1 + str(i) + '(normalized).npy'
    temp_data = np.concatenate((temp_data, np.load(path)), axis=0)
print("number of data is: " + str(np.shape(temp_data)[0]))
assert np.shape(temp_data)[0] == num_file * config.time, "Not reasonable data!!!"

intervals = config.intervals  # [1 day,20 min,143(label)]

#### apply data_loader to pre-process data that fit to our network
# use for test
processed_data = data_loader(temp_data, intervals)  # (4176,),c:(200, 200, 4),p:(200, 200, 4),l:(200, 200, 2)
train_data, test_data = create_train_test(processed_data)

batch_memory = np.array(test_data)[1]
c_input = Variable(torch.from_numpy(np.array([batch_memory.close]))).view(-1, 4, 200, 200).float().cuda()
p_input = Variable(torch.from_numpy(np.array([batch_memory.period]))).view(-1, 4, 200, 200).float().cuda()
e_input = Variable(torch.from_numpy(np.array([batch_memory.weather]))).float().cuda()
input = (c_input, p_input, None, e_input)

stnet = STResNet(external_dim=-1).cuda()

for i in range(43):
    a = 19 + i*20
    stnet.load_state_dict(torch.load(args.model_dir + 'model_{}.pth'.format(a)))
    main_output = stnet(input)
    main_output = main_output.cpu().data.squeeze().numpy()
    main_output = main_output*255

    f1, ax1 = plt.subplots()
    sns.heatmap(main_output[1], robust=False, ax=ax1)
    ax1.set_xlabel('')
    ax1.set_xticklabels([]) #设置x轴图例为空值
    ax1.set_yticklabels([]) #设置y轴图例为空值
    ax1.set_ylabel('')
    plt.savefig(args.result_dir+'outflow_{}_model'.format(a))

    print(i)

f2, ax2 = plt.subplots()
sns.heatmap(np.reshape(batch_memory.label, (2, 200, 200))[1]*255, robust=False, ax=ax2)
ax2.set_xlabel('')
ax2.set_xticklabels([]) #设置x轴图例为空值
ax2.set_yticklabels([]) #设置y轴图例为空值
ax2.set_ylabel('')
plt.savefig(args.result_dir+'outflow_{}'.format(859))