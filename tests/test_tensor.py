from pytensor.tensor import Tensor
from pytensor import operations


class TestTensor:
    def setup_method(self):
        self.a = Tensor([1, 2, 3], autograd=True)
        self.b = Tensor([4, 5, 6], autograd=True)

    def test_create(self):
        x = Tensor([1, 1, 1])
        assert all(x.data == [1, 1, 1])
        assert x.autograd is False

    def test_grad(self):
        a = Tensor([1, 2, 3], autograd=True)
        b = Tensor([4, 5, 6], autograd=True)
        c = Tensor([7, 8, 9], autograd=True)

        d = a + (-b)
        e = (-b) + c
        f = d + e

        f.backward(Tensor([1, 1, 1]))

        assert all(a.grad.data == [1, 1, 1])
        assert all(b.grad.data == [-2, -2, -2])
        assert all(c.grad.data == [1, 1, 1])
        assert all(d.grad.data == [1, 1, 1])
        assert all(e.grad.data == [1, 1, 1])
        assert all(f.grad.data == [1, 1, 1])

    def test_operation(self):
        t = self.a + self.b
        assert isinstance(t.op, operations.Add)
        t = -self.a
        assert isinstance(t.op, operations.Neg)
        t = self.a - self.b
        assert isinstance(t.op, operations.Sub)
        t = self.a * self.b
        assert isinstance(t.op, operations.Mul)
        t = self.a / self.b
        assert isinstance(t.op, operations.Truediv)
        t = self.a ** self.b
        assert isinstance(t.op, operations.Pow)
        t = self.a.matmul(self.b)
        assert isinstance(t.op, operations.MatMul)
        t = self.a.transpose()
        assert isinstance(t.op, operations.Transpose)
        t = self.a.sum(0)
        assert isinstance(t.op, operations.Sum)
        t = self.a.expand(0, 2)
        assert isinstance(t.op, operations.Expand)

    def test_zeros(self):
        x = Tensor.zeros(1)
        assert x.data == [0]
        x = Tensor.zeros(1, 2)
        assert all(x.data.all() == [[0], [0]])
