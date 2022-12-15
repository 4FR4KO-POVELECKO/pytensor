from pytensor.tensor import Tensor


class Optimizer(object):
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr

    def zero(self):
        for param in self.params:
            param.grad = None


class SGD(Optimizer):
    def step(self):
        for t in self.params:
            t.data -= t.grad.data * self.lr


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.001, eps=1e-08, decay=0.9):
        super().__init__(params, lr)
        self.epsilon, self.decay = eps, decay
        self.m_avg = 0

    def step(self):
        for t in self.params:
            self.m_avg = self.decay * self.m_avg + (1.0 - self.decay) * t.grad.data
            t.data -= (t.grad.data * self.lr) / (Tensor.sqrt(self.m_avg) + self.epsilon)
