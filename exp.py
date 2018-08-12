import numpy as np
import matplotlib.pyplot as plt
import os, imageio
from Config import *

args = GetConfiguration()
args.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) + '/SmartST/model_saved_rl/'
args.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)) + '/SmartST/result_saved_rl/'

if __name__ == '__main__':

    for i in range(100):
        a = np.random.randn(50, 1)
        _, _ = plt.subplots()
        plt.plot(a)
        plt.savefig(args.result_dir + 'test{}'.format(i) + '.png')
        print(i)

    plots = []

    for i in range(100):
        save_fn = args.result_dir + 'test{}'.format(i) + '.png'
        plots.append(imageio.imread(save_fn))
        print(i)

    imageio.mimsave(args.result_dir + 'test.gif', plots, fps=5)
