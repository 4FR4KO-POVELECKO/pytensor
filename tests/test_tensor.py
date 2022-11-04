from pytensor.tensor import Tensor
from example.first import FirstNN


class TestTensor:
    def setup_method(self):
        self.a = Tensor([1, 2, 3], autograd=True)
        self.b = Tensor([4, 5, 6], autograd=True)

    def test_add(self):
        x = self.a + self.b
        assert all(x.data == [5, 7, 9])

        x.backward(Tensor([1, 1, 1]))
        assert all(x.grad.data == [1, 1, 1])
        assert all(self.a.grad.data == [1, 1, 1])
        assert all(self.b.grad.data == [1, 1, 1])


    def test_neg(self):
        x = -self.a
        assert all(x.data == [-1, -2, -3])

        x.backward(Tensor([1, 1, 1]))
        assert all(x.grad.data == [1, 1, 1])
        assert all(self.a.grad.data == [-1, -1, -1])

    def test_sub(self):
        x = self.b - self.a
        assert all(x.data == [3, 3, 3])

        x.backward(Tensor([1, 1, 1]))
        assert all(x.grad.data == [1, 1, 1])
        assert all(self.a.grad.data == [-1, -1, -1])
        assert all(self.b.grad.data == [1, 1, 1])

    def test_mul(self):
        x = self.a * self.b
        assert all(x.data == [4, 10, 18])

        x.backward(Tensor([1, 1, 1]))
        assert all(x.grad.data == [1, 1, 1])
        assert all(self.a.grad.data == [4, 5, 6])
        assert all(self.b.grad.data == [1, 2, 3])

    def test_sum(self):
        x0 = Tensor([self.a.data, self.b.data], autograd=True).sum(0)
        x1 = Tensor([self.a.data, self.b.data], autograd=True).sum(1)
        assert all(x0.data == [5, 7, 9])
        assert all(x1.data == [6, 15])

    def test_expand(self):
        x = Tensor([self.a.data, self.b.data])
        x0 = x.expand(0, 2)
        x1 = x.expand(1, 2)
        assert (x0.data == [[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]]).all()
        assert (x1.data == [[[1,2,3], [1,2,3]], [[4,5,6], [4,5,6]]]).all()

    def test_transpose(self):
        x = Tensor([self.a.data, self.b.data]).transpose()
        assert (x.data == [[1, 4], [2, 5], [3, 6]]).all()

    def test_mmul(self):
        x = self.a.mmul(self.b)
        assert x.data == [32]

    # Test NN

    def test_first(self):
        nn = FirstNN()
        nn.train(output=False)
        assert nn.test() == True
