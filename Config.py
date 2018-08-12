###################################################################
#   Reinforcement Learning for Business
#   System Parameters Initial Settings

#   Torch Version: 0.3.1
#   Numpy Version: 1.14.0
#   Python Version: 3.6.4
###################################################################

import argparse

# for data_loader
time = 144
intervals = [0,137,143]

batch_size = 16

def GetConfiguration():
    parser = argparse.ArgumentParser(description='SmartST')

    parser.add_argument('--model-dir', metavar='DIR', help='path to data', default='/home/exx/Lab/SmartST/model_saved_rl/')
    parser.add_argument('--result-dir', metavar='DIR', help='path to data', default='/home/exx/Lab/SmartST/result_saved_rl/')
    parser.add_argument('--use-plt', metavar='PLT', default=False, type=bool, help='show image')

    parser.add_argument('--lrate', metavar='learning rate', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--GAMMA', metavar='gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--is-test', default=True, type=bool, help='test flag')

    args = parser.parse_args()

    return args
