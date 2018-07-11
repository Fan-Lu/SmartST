import paddle
import paddle.fluid as fluid
import numpy as np

from paddlemodel import STResNet

paddle.init(use_gpu=False, trainer_count=1)

def train(use_cuda, save_dirname, is_local):
    batch_size = 16
    epochs = 500

    c_input = paddle.layers.data(name='c_input', type=paddle.data_type.)
    p_input = paddle.layers.data(name='p_input')
    e_input = paddle.layers.data(name='e_input')

    label = paddle.layers.data(name='label')
    input = (c_input, p_input, None, e_input)
    main_output = STResNet(input)

    cost = fluid.layers.square_error_cost(main_output, label)
    optimizer =  fluid.optimizer.Adam(learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
    optimizer.minimize(cost)

    train_reader = paddle.batch(paddle.reader.shuffle(), batch_size=batch_size)

    if use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)


    def train_loop():
        feeder = fluid.DataFeeder(place=place, feed_list=[input, label])
        exe.run(fluid.default_startup_program())

        for epoch in range(epochs):
            for data in train_reader():
                loss = exe.run(main_program, feed=feeder.feed(data), fetch_list=[cost])

