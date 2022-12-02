from pytensor.tensor import Tensor
from pytensor import optimizers


class TestOptimizer:
    def setup_method(self):
        self.a = Tensor([1.0, 2.0, 3.0], autograd=True)
        self.b = Tensor([4.0, 5.0, 6.0], autograd=True)
        self.x = self.a + self.b
        self.x.backward(Tensor([1, 1, 1]))

    def test_sgd(self):
        optimizer = optimizers.SGD([self.x], lr=0.1)
        optimizer.step()
        assert all(self.x.data == [4.9, 6.9, 8.9])
        optimizer.zero()
        assert (self.x.grad is None)
