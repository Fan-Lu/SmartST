import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt


class STResNet(nn.Module):
    def __init__(self, c_conf=(2, 2, 32, 32), p_conf=(2, 2, 32, 32), t_conf=(2, 2, 32, 32), external_dim=8, nb_residual_unit=3):
        super(STResNet, self).__init__()
        '''
        C - Temporal Closeness
        P - Period
        T - Trend
        conf = (len_seq, nb_flow, map_height, map_width)
        external_dim
        '''
        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit

        len_seq, nb_flow, map_height, map_width = c_conf
        self.c_conv1 = nn.Conv2d(in_channels=nb_flow * len_seq, out_channels=64, kernel_size=3, padding=1)
        self.c_conv2 = nn.Conv2d(in_channels=64, out_channels=nb_flow, kernel_size=3, padding=1)

        len_seq, nb_flow, map_height, map_width = p_conf
        self.p_conv1 = nn.Conv2d(in_channels=nb_flow * len_seq, out_channels=64, kernel_size=3, padding=1)
        self.p_conv2 = nn.Conv2d(in_channels=64, out_channels=nb_flow, kernel_size=3, padding=1)

        # len_seq, nb_flow, map_height, map_width = t_conf
        # self.t_conv1 = nn.Conv2d(in_channels=nb_flow * len_seq, out_channels=64, kernel_size=3, padding=1)
        # self.t_conv2 = nn.Conv2d(in_channels=64, out_channels=nb_flow, kernel_size=3, padding=1)

        self.linear = nn.Linear(in_features=self.external_dim, out_features= 49)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=5)
        self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=5, stride=2)
        self.deconv5 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5, stride=3)
        self.deconv6 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4)
        self.relu = nn.ReLU()

        self.nb_flow = nb_flow
        self.map_height = map_height
        self.map_width = map_width

    def _shortcut(self, input, residual):
        return torch.add(input, residual)

    def _bn_relu_conv(self, nb_filter, ns_filter, bn=False):
        def f(input):
            if bn:
                input = nn.BatchNorm2d(input.size(1), affine=False).cuda()(input)
            activation = F.relu(input)
            return  nn.Conv2d(in_channels=input.size(1), out_channels=nb_filter, kernel_size=ns_filter, padding=1).cuda()(activation)
        return f

    def _residual_unit(self, nb_filter):
        def f(input):
            residual = self._bn_relu_conv(nb_filter=nb_filter, ns_filter=3)(input)
            residual = self._bn_relu_conv(nb_filter=nb_filter, ns_filter=3)(residual)
            return self._shortcut(input, residual)
        return f

    def ResUnits(self, residual_unit, nb_filter, repetations=1):
        def f(input):
            for i in range(repetations):
                input = residual_unit(nb_filter=nb_filter)(input)
            return input
        return f


    def forward(self, input):
        main_inputs = []
        outputs = []
        c_input, p_input, t_input, e_input = input
        main_inputs.append([c_input, p_input, t_input])

        #########c#########
        # conv1
        c_conv1 = self.c_conv1(c_input)
        c_residual_output = self.ResUnits(self._residual_unit, nb_filter=64, repetations=self.nb_residual_unit)(c_conv1)
        # conv2
        c_activation = F.relu(c_residual_output)
        c_conv2 = self.c_conv2(c_activation)
        outputs.append(c_conv2)
        #########p#########
        # conv1
        p_conv1 = self.p_conv1(p_input)
        p_residual_output = self.ResUnits(self._residual_unit, nb_filter=64, repetations=self.nb_residual_unit)(p_conv1)
        # conv2
        p_activation = F.relu(p_residual_output)
        p_conv2 = self.p_conv2(p_activation)
        outputs.append(p_conv2)
        #########t#########
        # conv1
        # t_conv1 = self.t_conv1(t_input)
        # t_residual_output = self.ResUnits(self._residual_unit, nb_filter=64, repetations=self.nb_residual_unit)(t_conv1)
        # # conv2
        # t_activation = F.relu(t_residual_output)
        # t_conv2 = self.c_conv2(t_activation)
        # outputs.append(t_conv2)

        # parameter-matrix-based fusion
        if len(outputs) == 1:
            main_output = outputs[0]
        else:
            from iLayer import iLayer
            new_outputs = []
            main_output = 0
            for output in outputs:
                cal = iLayer((output.size(1), output.size(2), output.size(3))).cuda()(output)
                new_outputs.append(cal)
                main_output += cal

        if self.external_dim != None and self.external_dim > 0:
            # external input
            main_inputs.append(e_input)

            e_output = self.relu(self.linear(e_input)).view(-1, 1, 7, 7)
            e_output = self.relu(self.deconv1(e_output))
            e_output = self.relu(self.deconv2(e_output))
            e_output = self.relu(self.deconv3(e_output))
            e_output = self.relu(self.deconv4(e_output))
            e_output = self.relu(self.deconv5(e_output))
            e_output = self.deconv6(e_output)

            main_output += e_output
        else:
            print('external_dim:', self.external_dim)

        main_output = F.tanh(main_output)

        return main_output

if __name__ == '__main__':
    print(1)
    # save_loss = []
    # # c_input, p_input, t_input, e_input
    # ground_truth = Variable(torch.randn(3, 2, 200, 200)).cuda()
    # c_input = Variable(torch.randn(3, 4, 200, 200)).cuda()
    # p_input = Variable(torch.randn(3, 4, 200, 200)).cuda()
    # # t_input = Variable(torch.randn(3, 6, 200, 200)).cuda()
    # e_input = Variable(torch.randn(3, 2, 200, 200)).cuda() # uncertain variable
    # for i in range(1000):
    #     input = (c_input, p_input, None, e_input)
    #     main_output = stnet(input)
    #     optimizer.zero_grad()
    #     loss = criterion(main_output, ground_truth)
    #     loss.backward()
    #     optimizer.step()
    #     save_loss.append(loss.cpu().data.numpy())
    #     print(i)
    # # plt.switch_backend('agg')
    # plt.plot(save_loss)
    # plt.show()
    # plt.savefig('{}{}'.format('/mnt/data/fan/SmartST', 'loss'))
