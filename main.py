import os
import Config as config
import numpy as np
from Utils import data_loader
from model import STResNet
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='SmartST')
parser.add_argument('--model-dir', metavar='DIR', help='path to data', default='/home/exx/Lab/SmartST/model_saved/')
parser.add_argument('--result-dir', metavar='DIR', help='path to data', default='/home/exx/Lab/SmartST/result_saved/')
parser.add_argument('--use-plt', default=False, type=bool, help='plot figure')
args = parser.parse_args()

stnet = STResNet(external_dim=-1).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(stnet.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

if __name__ == '__main__':

    #### load original data
    pa_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) # get last state file name
    name1 = '/SmartST/data/data(normalized)/'
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
    processed_data = data_loader(temp_data,intervals) # (4176,),c:(200, 200, 4),p:(200, 200, 4),l:(200, 200, 2)

    for epoch in range(100):
        save_loss = []
        for i in range(4176):
            c_input, l_input, p_input = processed_data[i].close, processed_data[i].label, processed_data[i].period
            c_input = Variable(torch.from_numpy(c_input)).view(-1, 4, 200, 200).float().cuda()
            p_input = Variable(torch.from_numpy(p_input)).view(-1, 4, 200, 200).float().cuda()
            l_input = Variable(torch.from_numpy(l_input)).view(-1, 2, 200, 200).float().cuda()  # label

            input = (c_input, p_input, None, None)
            main_output = stnet(input)
            optimizer.zero_grad()
            loss = criterion(main_output, l_input)
            loss.backward()
            optimizer.step()
            save_loss.append(loss.cpu().data.numpy()[0])
            print('epoch: 1/{}, iter: 4176/{}, loss: {}'.format(epoch, i, loss.cpu().data.numpy()[0]))

        if (epoch) % 20 == 0:
            if not os.path.exists(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(stnet.state_dict(), args.model_dir + 'model_{:d}.pth'.format(epoch))

            if not os.path.exists(args.result_dir):
                os.mkdir(args.result_dir)
            if not args.use_plt: plt.switch_backend('agg')
            plt.figure('Learning Curve')
            plt.xlabel('Iteration')
            plt.ylabel('MSE Loss')
            plt.plot(save_loss, 'r')
            plt.savefig(args.result_dir + 'train_result_epoch_{:d}'.format(epoch))
            if args.use_plt: plt.show()
