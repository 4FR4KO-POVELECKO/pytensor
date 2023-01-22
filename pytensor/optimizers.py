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
    def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
        super().__init__(params, lr)
        self.decay, self.epsilon = decay, eps
        self.m_avg = [Tensor.zeros(*t.shape) for t in self.params]

    def step(self):
        for i, t in enumerate(self.params):
            self.m_avg[i] = self.decay * self.m_avg[i] + (1.0 - self.decay) * t.grad.data
            t.data -= (t.grad.data * self.lr) / (Tensor.np.sqrt(self.m_avg[i]) + self.epsilon)


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta_1, self.beta_2, self.epsilon = b1, b2, eps
        self.m = [Tensor.zeros(*t.shape) for t in self.params]
        self.v = [Tensor.zeros(*t.shape) for t in self.params]

    def step(self):
        for i, t in enumerate(self.params):
            self.m[i] = self.beta_1 * self.m[i] + (1.0 - self.beta_1) * t.grad.data
            self.v[i] = self.beta_2 * self.v[i] + (1.0 - self.beta_2) * Tensor.np.sqrt(t.grad.data)

            m_hat = self.m[i] / (1.0 - self.beta_1 ** i+1)
            v_hat = self.v[i] / (1.0 - self.beta_2 ** i+1)

            t.data -= self.lr * m_hat / (Tensor.np.sqrt(v_hat) + self.epsilon)
