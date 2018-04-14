'''
Created on 5 de dez de 2017

@author: marcelovca90
'''
import numpy as np
from util import DataUtils
from util import MathUtils
from util import PlotUtils
from data import SampleData

class SingleLayerPerceptronDelta:

    def __init__(self):
        self.n = 1e-3 # learning rate
        self.e = 1e-3 # error threshold
        self.g = MathUtils.sign # activation function
        self.plot_data_x = [] # epochs for plotting
        self.plot_data_y = [] # eqms for plotting

    def train(self, x, d):
        k = len(x)
        w = np.random.rand(len(x[0]))
        epoch = 0
        while True:
            eqm_prev = MathUtils.eqm(w, x, d)
            for i in range(0, k):
                v = np.dot(np.transpose(w), x[i])
                w = np.add(w, np.multiply(x[i], self.n * (d[i] - v)))
            epoch = epoch + 1
            eqm_curr = MathUtils.eqm(w, x, d)
            eqm_delta = abs(eqm_curr - eqm_prev)
            print('epoch = {}\tw = {}\teqm(abs) = {}'.format(epoch, w, eqm_delta))
            self.plot_data_x.append(epoch)
            self.plot_data_y.append(eqm_delta)
            if eqm_delta < self.e:
                break
        return w
            
    def test(self, w, x):
        v = np.dot(np.transpose(w), x)
        y = self.g(v)
        return y;

if  __name__ == '__main__':
    
    # set random number generator seed
    np.random.seed(DataUtils.random_seed())
    
    # set floating point formatting when printing
    np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
    
    # load data
    x = SampleData.TIC_TAC_TOE_ENDGAME.input
    d = SampleData.TIC_TAC_TOE_ENDGAME.output
    
    # prepare data
    x = DataUtils.add_bias(x)
    x,d = DataUtils.shuffle(x,d)
    x_train,x_test = DataUtils.split(x)
    d_train,d_test = DataUtils.split(d)
    
    # create the neural network
    nn = SingleLayerPerceptronDelta()
    
    # train the neural network
    w = nn.train(x_train, d_train)
    
    # plot epoch versus eqm data
    PlotUtils.plot(nn.plot_data_x, 'epoch', nn.plot_data_y, 'eqm(abs)', nn.e)
    
    # test the neural network
    correct = 0
    for i in range(0, len(x_test)):
        y = nn.test(w, x_test[i])
        if (y == d_test[i]):
            correct = correct + 1
    accuracy = 100.0 * float(correct) / float(len(x_test))
    print('accuracy: {}/{} ({:.2f}%)'.format(correct, len(x_test), accuracy))
