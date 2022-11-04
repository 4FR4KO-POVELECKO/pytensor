from pytensor.tensor import Tensor

import numpy as np
np.random.seed(0)


class FirstNN:
    def __init__(self):
        self.x_train = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
        self.y_train = Tensor([[0], [1], [0], [1]], autograd=True)

        self.w = list()
        self.w.append(Tensor(np.random.rand(2, 3), autograd=True))
        self.w.append(Tensor(np.random.rand(3, 1), autograd=True))

    @property
    def prediction(self):
        return self.x_train.mmul(self.w[0]).mmul(self.w[1])

    @property
    def loss(self):
        return ((self.prediction - self.y_train) * (self.prediction - self.y_train)).sum(0)

    def train(self, epochs=10, alpha=0.1, output=True):
        for i in range(epochs):
            self.loss.backward(Tensor(np.ones_like(self.loss.data)))

            for w_ in self.w:
                w_.data -= w_.grad.data * alpha
                w_.grad.data *= 0

            if output:
                print(f'Epoch: {i}. Loss: {self.loss}.')

    def test(self):
        return all(np.around(self.prediction.data) == self.y_train.data)


if __name__ == '__main__':
    nn = FirstNN()
    nn.train()
    nn.test()
