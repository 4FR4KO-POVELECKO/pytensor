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
        test_x = self.x.data - (self.x.grad.data * optimizer.lr) / (Tensor.np.sqrt(m_avg) + optimizer.epsilon)
        optimizer.step()
        assert all(self.x.data == test_x)

    def test_adam(self):
        optimizer = optimizers.Adam([self.x], lr=0.1)
        m = optimizer.beta_1 * 0 + (1.0 - optimizer.beta_1) * self.x.grad.data
        v = optimizer.beta_2 * 0 + (1.0 - optimizer.beta_2) * Tensor.np.sqrt(self.x.grad.data)
        m_hat = m / (1.0 - optimizer.beta_1 ** 0+1)
        v_hat = v / (1.0 - optimizer.beta_2 ** 0+1)
        test_x = self.x.data - (optimizer.lr * m_hat / (Tensor.np.sqrt(v_hat) + optimizer.epsilon))
        optimizer.step()
        assert all(self.x.data == test_x)
