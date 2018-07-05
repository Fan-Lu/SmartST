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
            embedding = F.relu(nn.Linear(in_features=e_input.size(1), out_features=10).cuda()(e_input))
            h1 = F.relu(nn.Linear(in_features=10, out_features=self.nb_flow * self.map_height * self.map_width).cuda()(embedding))
            e_output = h1.view(-1, self.nb_flow, self.map_height, self.map_width)
            main_output += e_output
        # TODO: Add External Info
        # else:
            # print('external_dim:', self.external_dim)


        main_output = F.tanh(main_output)

        return main_output

stnet = STResNet(external_dim=-1).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(stnet.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)


if __name__ == '__main__':
    save_loss = []
    # c_input, p_input, t_input, e_input
    ground_truth = Variable(torch.randn(3, 2, 200, 200)).cuda()
    c_input = Variable(torch.randn(3, 4, 200, 200)).cuda()
    p_input = Variable(torch.randn(3, 4, 200, 200)).cuda()
    # t_input = Variable(torch.randn(3, 6, 200, 200)).cuda()
    e_input = Variable(torch.randn(3, 2, 200, 200)).cuda() # uncertain variable
    for i in range(1000):
        input = (c_input, p_input, None, e_input)
        main_output = stnet(input)
        optimizer.zero_grad()
        loss = criterion(main_output, ground_truth)
        loss.backward()
        optimizer.step()
        save_loss.append(loss.cpu().data.numpy())
        print(i)
    # plt.switch_backend('agg')
    plt.plot(save_loss)
    plt.show()
    # plt.savefig('{}{}'.format('/mnt/data/fan/SmartST', 'loss'))
