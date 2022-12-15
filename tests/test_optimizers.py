from pytensor.tensor import Tensor
from pytensor import optimizers


class TestOptimizer:
    def setup_method(self):
        self.a = Tensor([1.0, 2.0, 3.0], autograd=True)
        self.b = Tensor([4.0, 5.0, 6.0], autograd=True)
        self.x = self.a + self.b
        self.x.backward(Tensor([1, 1, 1]))

    def test_zero(self):
        optimizer = optimizers.Optimizer([self.x])
        assert (self.x.grad is not None)
        optimizer.zero()
        assert (self.x.grad is None)

    def test_sgd(self):
        optimizer = optimizers.SGD([self.x], lr=0.1)
        test_x = self.x.data - self.x.grad.data * optimizer.lr
        optimizer.step()
        assert all(self.x.data == test_x)

    def test_rmsprop(self):
        optimizer = optimizers.RMSprop([self.x], lr=0.1)
        m_avg = optimizer.decay * 0 + (1.0 - optimizer.decay) * self.x.grad.data
        test_x = self.x.data - (self.x.grad.data * optimizer.lr) / (Tensor.sqrt(m_avg) + optimizer.epsilon)
        optimizer.step()
        assert all(self.x.data == test_x)
