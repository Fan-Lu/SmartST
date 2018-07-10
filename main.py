import os
import Config as config
import numpy as np
from Utils import *
from model import STResNet
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

parser = argparse.ArgumentParser(description='SmartST')
parser.add_argument('--model-dir', metavar='DIR', help='path to data', default='/mnt/data//fan/SmartST/model_saved/')
parser.add_argument('--result-dir', metavar='DIR', help='path to data', default='/mnt/data//fan/SmartST/result_saved/')
# parser.add_argument('--model-dir', metavar='DIR', help='path to data', default='/home/exx/Lab/SmartST/model_saved/')
# parser.add_argument('--result-dir', metavar='DIR', help='path to data', default='/home/exx/Lab/SmartST/result_saved/')
parser.add_argument('--use-plt', default=False, type=bool, help='plot figure')
parser.add_argument('--batch-size', default=16, type=int, help='batch size default=32')
args = parser.parse_args()

stnet = STResNet(external_dim=9).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(stnet.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)


if __name__ == '__main__':

    #### load original data
    pa_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) # get last state file name
    name1 = '/SmartST/data/data(normalized)/'
    name_start = 20161101
    num_file = 3
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
    processed_data = data_loader(temp_data, intervals) # (4176,),c:(200, 200, 4),p:(200, 200, 4),l:(200, 200, 2)
    train_data, test_data = create_train_test(processed_data)

    train_size = train_data.__len__()
    iter_size = int(train_size/args.batch_size)
    train_loss_save = []
    test_loss_save = []
    for epoch in range(1000):
        sample_index = np.random.shuffle(train_data)
        move_loss = 0
        for i in range(iter_size):
            # sample_index = np.random.shuffle(processed_data)
            batch_memory = np.array(train_data)[i*args.batch_size:(i+1)*args.batch_size]

            c_input = Variable(torch.from_numpy(np.array([memory_unit.close for memory_unit in batch_memory]))).view(-1, 4, 200, 200).float().cuda()
            p_input = Variable(torch.from_numpy(np.array([memory_unit.period for memory_unit in batch_memory]))).view(-1, 4, 200, 200).float().cuda()
            l_input = Variable(torch.from_numpy(np.array([memory_unit.label for memory_unit in batch_memory]))).view(-1, 2, 200, 200).float().cuda()
            e_input = Variable(torch.from_numpy(np.array([memory_unit.weather for memory_unit in batch_memory]))).float().cuda()

            input = (c_input, p_input, None, e_input)
            main_output = stnet(input)
            optimizer.zero_grad()
            loss = criterion(main_output, l_input)
            loss.backward()
            optimizer.step()
            move_loss += loss.cpu().data.numpy()[0]
            print('Training Epoch: {}/1000, Iter: {}/{}, Loss: {}'.format(epoch+1, i+1, iter_size, move_loss/(i+1)))
        train_loss_save.append(move_loss/iter_size)
        # run test every 20 epochs and save results and models
        if epoch % 20 == 19:
            if not os.path.exists(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(stnet.state_dict(), args.model_dir + 'model_{:d}.pth'.format(epoch))
            move_test_loss = 0
            test_iter = int(test_data.__len__()/args.batch_size)
            for i in range(test_iter):
                batch_memory = np.array(test_data)[i * args.batch_size:(i + 1) * args.batch_size]
                c_input = Variable(torch.from_numpy(np.array([memory_unit.close for memory_unit in batch_memory]))).view(-1, 4, 200, 200).float().cuda()
                p_input = Variable(torch.from_numpy(np.array([memory_unit.period for memory_unit in batch_memory]))).view(-1, 4, 200, 200).float().cuda()
                l_input = Variable(torch.from_numpy(np.array([memory_unit.label for memory_unit in batch_memory]))).view(-1, 2, 200, 200).float().cuda()
                e_input = Variable(torch.from_numpy(np.array([memory_unit.weather for memory_unit in batch_memory]))).float().cuda()
                input = (c_input, p_input, None, e_input)
                main_output = stnet(input)
                test_loss = criterion(main_output, l_input)
                move_test_loss += test_loss.cpu().data.numpy()[0]
                print('Testing Epoch: {}/1000, Iter: {}/{}, Loss: {}'.format(epoch + 1, i + 1, test_iter, move_test_loss / (i + 1)))

            test_loss_save.append(move_test_loss/(test_data.__len__()+1))

            if not os.path.exists(args.result_dir):
                os.mkdir(args.result_dir)
            if not args.use_plt: plt.switch_backend('agg')
            plt.figure('Learning Curve')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            x_train = np.arange(epoch+1)
            x_test = np.arange(19, epoch + 1, 20)
            p1 = plt.plot(x_train, train_loss_save, 'r')
            p2 = plt.plot(x_test, test_loss_save, 'b')
            plt.legend([p1[0], p2[0]], ['Train', 'Test'])
            plt.savefig(args.result_dir + 'train_test_result_epoch_{:d}'.format(epoch))
            if args.use_plt: plt.show()

            train_loss_data = pd.DataFrame(train_loss_save)
            train_loss_data.to_csv(args.result_dir + 'train_loss.csv')

            test_loss_data = pd.DataFrame(test_loss_save)
            test_loss_data.to_csv(args.result_dir + 'test_loss.csv')
    print('All Finished')
