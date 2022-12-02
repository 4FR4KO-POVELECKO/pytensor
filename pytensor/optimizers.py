class Optimizer(object):
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr

    def zero(self):
        for param in self.params:
            param.grad = None


class SGD(Optimizer):
    def step(self):
        for t in self.params:
            t.data -= t.grad.data * self.lr
