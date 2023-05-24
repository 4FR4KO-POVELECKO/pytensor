from pytensor.tensor import Tensor
from pytensor import operations


class TestOperation:
    def setup_method(self):
        self.a = Tensor([1, 2, 3], autograd=True)
        self.b = Tensor([4, 5, 6], autograd=True)

    def round_array(self, x):
        for i, v in enumerate(x):
            x[i] = round(v, 2)
        return x

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

    def test_truediv(self):
        operation = operations.Truediv(self.a, self.b)

        f = operation.forward(self.a.data, self.b.data)
        assert all(f.data == [0.25, 0.4, 0.5])

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.round_array(self.a.grad.data) == [0.25, 0.20, 0.17])
        assert all(self.round_array(self.b.grad.data) == [-0.06, -0.08, -0.08])

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

    def test_pow(self):
        operation = operations.Pow(self.a, self.b)

        f = operation.forward(self.a.data, self.b.data)
        assert all(f.data == [1, 32, 729])

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.a.grad.data == [4, 80, 1458])
        assert all(self.round_array(self.b.grad.data) == [0, 22.18, 800.89])

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

    def test_sigmoid(self):
        operation = operations.Sigmoid(self.a)

        f = operation.forward(self.a)
        assert all(self.round_array(f.data) == [0.98, 1., 1.])

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.round_array(self.a.grad.data) == [0.02, 0., 0.])

    def test_tanh(self):
        operation = operations.Tanh(self.a)

        f = operation.forward(self.a)
        assert all(self.round_array(f.data) == [0.76, 0.96, 1.])

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.round_array(self.a.grad.data) == [0.42, 0.08, 0.])

    def test_softmax(self):
        operation = operations.Softmax(self.a)

        f = operation.forward(self.a)
        assert all(self.round_array(f.data) == [0.09, 0.24, 0.67])

        operation.backward(Tensor([1, 1, 1]))
        assert self.a.grad == [-1]

    def test_cross_entropy(self):
        operation = operations.CrossEntropy(self.a)
        target = Tensor([1])

        f = operation.forward(self.a, target.data)
        assert f.data == [1.407605555828337]

        operation.backward(Tensor([1, 1, 1]))
        assert all(self.round_array(self.a.grad.data[0]) == [0.09, -0.76, 0.67])
