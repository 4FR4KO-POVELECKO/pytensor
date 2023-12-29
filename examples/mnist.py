#!/usr/bin/env python3
import __init__
import numpy as np
from pytensor.tensor import Tensor
from pytensor.layers import Sequential, Linear, MSELoss
from pytensor.optimizers import SGD


path = './datasets/mnist/mnist.npz'


class NN:
    def __init__(self):
        # (self.x_train, self.y_train), (self.x_test, self.y_test) = None, None, None, None # load your mnist data
        with np.load(path, allow_pickle=True) as f:
            self.x_train, self.y_train = f["x_train"], f["y_train"]
            self.x_test, self.y_test = f["x_test"], f["y_test"]

        self.x_train = Tensor(self.x_train.reshape(60000, 784), autograd=True)
        self.y_train = Tensor(self.y_train, autograd=True)
        # assert self.x_train.shape == (60000, 784)
        # assert self.y_train.shape == (60000,)
        
        # self.x_test = Tensor(self.x_test.reshape(10000, 784), autograd=True)
        # self.y_test = Tensor(self.y_test, autograd=True)
        # assert self.x_test.shape == (10000, 784)
        # assert self.y_test.shape == (10000,)

        self.model = Sequential()
        self.model.add(Linear(784, 128, bias=True))
        self.model.add(Linear(128, 10, bias=True))

        self.optimizer = SGD(params=self.model.get_params())
        self.criterion = MSELoss()

    @property
    def prediction(self):
        return self.model.forward(self.x_train)

    def train(self, epochs=10, alpha=0.01, output=True):
        self.optimizer.lr = alpha
        
        for idx, X in enumerate(self.x_train):
            # print(X.shape)
            pred = self.model.forward(Tensor(X))
            pred.data = Tensor.np.argmax(pred, axis=0)
            print(pred)
            # print(type(pred))
            # pred.data = Tensor.np.argmax(pred)
            # print(pred.data)
            y = Tensor.zeros(10)
            y[self.y_train[idx]-1] = 1
            print(y)
            # print(self.y_train[idx])
            loss = self.criterion.forward(pred, y)
            print(loss)
            # print(loss)
            # print(type(loss))
            # ones = Tensor.ones_like(loss)
            # print(ones)
            # print(type(ones))
            # print(ones.shape)
            # print(loss.shape)
            loss.backward(Tensor.ones_like(loss))
            self.optimizer.step()
            break

            if output:
                pass
                # print(f'Epoch: {e}. Loss: {loss}.')
    

if __name__ == '__main__':
    nn = NN()
    nn.train()
    # nn.test()
