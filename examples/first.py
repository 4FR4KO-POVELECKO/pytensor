#!/usr/bin/env python3
import numpy as np
from pytensor.tensor import Tensor
from pytensor.layers import Sequential, Linear, MSELoss, Sigmoid, Tanh
from pytensor import optimizers


class FirstNN:
    def __init__(self):
        self.x_train = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
        self.y_train = Tensor([[0], [1], [0], [1]], autograd=True)

        self.model = Sequential()
        self.model.add(Linear(2, 3, bias=True))
        self.model.add(Tanh())
        self.model.add(Linear(3, 1, bias=True))
        self.model.add(Sigmoid())

        self.optimizer = optimizers.SGD(params=self.model.get_params())
        self.criterion = MSELoss()

    @property
    def prediction(self):
        return self.model.forward(self.x_train)

    @property
    def loss(self):
        return self.criterion.forward(self.prediction, self.y_train)

    def train(self, epochs=10, alpha=1, output=True):
        self.optimizer.lr = alpha
        for i in range(epochs):
            self.loss.backward(Tensor.ones_like(self.loss.data))
            self.optimizer.step()
            self.optimizer.zero()

            if output:
                print(f'Epoch: {i}. Loss: {self.loss}.')

    def test(self):
        return all(np.around(self.prediction.data) == self.y_train.data)


if __name__ == '__main__':
    nn = FirstNN()
    nn.train()
    nn.test()
