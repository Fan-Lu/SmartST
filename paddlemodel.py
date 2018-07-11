import paddle
import paddle.fluid as fluid

def STResNet(input, c_conf=(2, 2, 32, 32), p_conf=(2, 2, 32, 32), external_dim=8, nb_residual_unit=3):
    def _shortcut(input, residual):
        return fluid.layers.elementwise_add(input, residual)

    def _bn_relu_conv(input, nb_filter, ns_filter, bn=False):
        if bn:
            input = fluid.layers.batch_norm(input)
        activation = fluid.layers.relu(input)
        return fluid.layers.conv2d(input=activation, num_filters=nb_filter,filter_size=ns_filter,padding=1)

    def _residual_unit(input, nb_filter):
        residual = _bn_relu_conv(input, nb_filter=nb_filter, ns_filter=3)
        residual = _bn_relu_conv(residual, nb_filter=nb_filter, ns_filter=3)
        return _shortcut(input, residual)

    def ResUnits(input, nb_filter, repetations=1):
        for i in range(repetations):
            input = _residual_unit(input ,nb_filter=nb_filter)
        return input


    c_input, p_input, _, e_input = input
    len_seq, nb_flow, map_height, map_width = c_conf
    len_seq, nb_flow, map_height, map_width = p_conf

    c_conv1 = fluid.layers.conv2d(input=input, num_filters=64, filter_size=3, padding=1)


    main_output =
    return main_output