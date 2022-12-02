from pytensor.tensor import Tensor
from pytensor import operations


class TestOperation:
    def setup_method(self):
        self.a = Tensor([1, 2, 3], autograd=True)
        self.b = Tensor([4, 5, 6], autograd=True)

    def test_add(self):
        operation = operations.Add(self.a, self.b)

        f = operation.forward(self.a.data, self.b.data)
        assert all(f.data == [5, 7, 9])

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.a.grad.data == [1, 1, 1])
        assert all(self.b.grad.data == [1, 1, 1])

    def test_neg(self):
        operation = operations.Neg(self.a)

        f = operation.forward(self.a.data)
        assert all(f.data == [-1, -2, -3])

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.a.grad.data == [-1, -1, -1])

    def test_sub(self):
        operation = operations.Sub(self.a, self.b)

        f = operation.forward(self.b.data, self.a.data)
        assert all(f.data == [3, 3, 3])

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.a.grad.data == [1, 1, 1])
        assert all(self.b.grad.data == [-1, -1, -1])

    def test_mul(self):
        operation = operations.Mul(self.a, self.b)

        f = operation.forward(self.a.data, self.b.data)
        assert all(f.data == [4, 10, 18])

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.a.grad.data == [4, 5, 6])
        assert all(self.b.grad.data == [1, 2, 3])

    def test_matmul(self):
        operation = operations.MatMul(self.a, self.b)

        f = operation.forward(self.a.data, self.b.data)
        assert all(f.data == [32])

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.a.grad.data == [15])
        assert all(self.b.grad.data == [6])

    def test_transpose(self):
        x = Tensor([self.a.data, self.b.data], autograd=True)
        operation = operations.Transpose(x)

        f = operation.forward(x.data)
        assert (f.data == [[1, 4], [2, 5], [3, 6]]).all()

        operation.backward(Tensor([[1, 1, 1], [2, 2, 2]]))
        assert (x.grad.data == [[1, 2], [1, 2], [1, 2]]).all()

    def test_sum(self):
        x = Tensor([self.a.data, self.b.data], autograd=True)
        operation = operations.Sum(x)

        f = operation.forward(x.data, 0)
        assert all(f.data == [5, 7, 9])
        f = operation.forward(x.data, 1)
        assert all(f.data == [6, 15])

        operation.backward(Tensor([1, 1, 1]))
        assert (x.grad.data == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]).all()

    def test_expand(self):
        x = Tensor([self.a.data, self.b.data], autograd=True)
        operation = operations.Expand(x)

        f = operation.forward(x.data, 0, 2)
        assert (f.data == [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]).all()

        f = operation.forward(x.data, 1, 2)
        assert (f.data == [[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6]]]).all()

        operation.backward(Tensor([[1, 1, 1], [2, 2, 2]]))
        assert all(x.grad.data == [3, 6])
