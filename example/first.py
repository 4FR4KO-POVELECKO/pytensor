from pytensor.tensor import Tensor
from pytensor.layers import Sequential, Linear
from pytensor import optimizers
import numpy as np
np.random.seed(0)


class FirstNN:
    def __init__(self):
        self.x_train = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
        self.y_train = Tensor([[0], [1], [0], [1]], autograd=True)

        self.model = Sequential()
        self.model.add(Linear(2, 3, bias=True))
        self.model.add(Linear(3, 1, bias=True))
        self.optimizer = optimizers.SGD(params=self.model.get_params())

    @property
    def prediction(self):
        return self.model.forward(self.x_train)

    @property
    def loss(self):
        return ((self.prediction - self.y_train) * (self.prediction - self.y_train)).sum(0)

    def train(self, epochs=10, alpha=0.1, output=True):
        self.optimizer.lr = alpha
        for i in range(epochs):
            self.loss.backward(Tensor(np.ones_like(self.loss.data)))
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
