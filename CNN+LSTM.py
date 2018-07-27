import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import numpy as np
from Utils import *

batch_size = 16
epochs = 50

c_input = Variable(torch.FloatTensor(1, 2, 200, 200)).cuda()
l_input = Variable(torch.FloatTensor(2, 2, 200, 200))
e_input = Variable(torch.FloatTensor(16, 9))

class Autodecoder(nn.Module):
    def __init__(self):
        super(Autodecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=2, bias=True)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=2, bias=True)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, bias=True)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, bias=True)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, bias=True)
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, bias=True)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.relu = nn.ReLU()
        self.deconv7 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, bias=True)
        self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, bias=True)
        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, bias=True)
        self.deconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, bias=True)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=6, stride=2, bias=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=6, stride=2, bias=True)
        self.deconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=5, stride=1, bias=True)

    def forward(self, input):
        input = self.relu(self.conv1(input))
        input = self.relu(self.conv2(input))
        input = self.relu(self.conv3(input))
        input = self.relu(self.conv4(input))
        input = self.relu(self.conv5(input))
        input = self.relu(self.conv6(input))
        input = self.relu(self.conv7(input))

        input = self.relu(self.fc1(input.view(batch_size, -1))).view(batch_size, -1, 2, 2)

        input = self.relu(self.deconv7(input))
        input = self.relu(self.deconv6(input))
        input = self.relu(self.deconv5(input))
        input = self.relu(self.deconv4(input))
        input = self.relu(self.deconv3(input))
        input = self.relu(self.deconv2(input))
        input = self.deconv1(input)
        return input



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
processed_data = data_loader(temp_data, intervals) #  (4176,),c:(200, 200, 4),p:(200, 200, 4),l:(200, 200, 2)
train_data, test_data = create_train_test(processed_data)

train_size = train_data.__len__()
iter_size = int(train_size/batch_size)
train_loss = []
test_loss = []
model = Autodecoder().cuda()
Loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(epochs):
    np.random.shuffle(train_data)
    moving_loss = 0
    counter = 0
    for i in range(iter_size):
        model.train(True)
        batch_memory = np.array(train_data)[i * batch_size:(i + 1) * batch_size]
        label = Variable(torch.from_numpy(np.array([memory_unit.label for memory_unit in batch_memory]))).view(-1, 2, 200, 200).float().cuda()
        output = model(label)
        optimizer.zero_grad()
        loss = Loss(output, label)
        loss.backward()
        optimizer.step()
        moving_loss += loss.data
        counter += 1
        if i % 10 == 0:
            print('Training Epoch: {}, iteration: {}, loss: {}'.format(epoch, i, loss.data))
        if i % 50 == 0:
            moving_loss = 0
            train_loss.append(moving_loss/counter)
            counter = 0
            with torch.no_grad():
                model.train(False)
                test_moving_loss = 0
                torch.save(model.state_dict(), 'params_{}.pkl'.format(epoch))
                test_iter = int(test_data.__len__()/batch_size)
                for j in range(test_iter):
                    test_batch_memory = test_data[j * batch_size: (j+1) * batch_size]
                    test_label = Variable(torch.from_numpy(np.array([unit.label for unit in test_batch_memory]))).view(-1, 2, 200, 200).float().cuda()
                    test_out = model(test_label)
                    te_loss = Loss(test_out, test_label)
                    test_moving_loss += te_loss.data
                test_loss.append(test_moving_loss/test_iter)
                print('Test Epoch: {}, iteration: {}, test loss: {}'.format(epoch, i, te_loss.data))

plt.switch_backend('agg')
plt.figure('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
x_train = np.arange(len(train_loss))
p1 = plt.plot(x_train, train_loss, 'r')
p2 = plt.plot(x_train, test_loss, 'b')
plt.legend([p1[0], p2[0]], ['Train', 'Test'])
plt.savefig('train_test_result_epoch_{:d}'.format(epochs))
print('All Finished')





