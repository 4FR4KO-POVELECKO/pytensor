import inspect
import importlib
import numpy as np


class Tensor(object):
    def __init__(self, data: list, autograd: bool = False):
        """
        :param data(list): np.array or list
        :param autograd(bool): default False
        """
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.autograd = autograd
        self.grad: Tensor = None
        self.op: Operation = None

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape), **kwargs)

    @classmethod
    def zeros_like(cls, *shape, **kwargs):
        return cls(np.zeros_like(*shape), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.rand(*shape), **kwargs)

    @classmethod
    def ones_like(cls, *shape, **kwargs):
        return cls(np.ones_like(*shape), **kwargs)

    def assign(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        assert self.shape == x.shape
        self.data = x.data
        return x

    def backward(self, grad):
        if not self.autograd:
            return
        self.grad = grad if not self.grad else self.grad + grad

        if self.op and self.op.parents:
            self.op.backward(grad)

    def _execute(self, operation, *data):
        tensors = [tensor for tensor in data if type(tensor) is Tensor]  # only tensors
        op = operation(*tensors)
        return op.forward(*[d.data if type(d) is Tensor else d for d in data])

    def sum(self, dim):
        return self._execute(Tensor._sum, self, dim)

    def expand(self, dim, copies):
        return self._execute(Tensor._expand, self, dim, copies)

    def transpose(self):
        return self._execute(Tensor._transpose, self)

    def matmul(self, x):
        return self._execute(Tensor._matmul, self, x)

    def sigmoid(self):
        return self._execute(Tensor._sigmoid, self)

    def tanh(self):
        return self._execute(Tensor._tanh, self)

    def softmax(self):
        return self._execute(Tensor._softmax, self)

    def add(self, x): return self.__add__(x)
    def sub(self, x): return self.__sub__(x)
    def mul(self, x): return self.__mul__(x)
    def pow(self, x): return self.__pow__(x)
    def truediv(self, x): return self.__truediv__(x)
    def neg(self): return self.__neg__()

    def __add__(self, x):
        return self._execute(Tensor._add, self, x)

    def __sub__(self, x):
        return self._execute(Tensor._sub, self, x)

    def __mul__(self, x):
        return self._execute(Tensor._mul, self, x)

    def __pow__(self, x):
        return self._execute(Tensor._pow, self, x)

    def __truediv__(self, x):
        return self._execute(Tensor._truediv, self, x)

    def __neg__(self):
        return self._execute(Tensor._neg, self)

    def __repr__(self): return str(self.data.__repr__())
    def __str__(self): return str(self.data.__str__())

    def __len__(self): return len(self.data)

    def getitem(self, x):
        x = x.data if isinstance(x, Tensor) else x
        return self._execute(Tensor._getitem, self, x)

    def cross_entropy(self, x):
        x = x.data if isinstance(x, Tensor) else x
        return self._execute(Tensor._crossentropy, self, x)

    def __setitem__(self, x, value): self.data[x] = value
    def __getitem__(self, x): return self.data[x]


class Operation(object):
    def __init__(self, *tensors: Tensor):
        self.parents = tensors
        self.autograd = any([tensor.autograd for tensor in tensors])
        self.new = None

    def forward(self, *args):
        raise NotImplementedError()

    def backward(self, grad: Tensor):
        raise NotImplementedError()

    def new_tensor(self, data):
        tensor = Tensor(data, autograd=self.autograd)
        tensor.op = self
        self.new = tensor
        return tensor


# Set ops
# operation.Add -> Tensor._add ...
for name, cls in inspect.getmembers(importlib.import_module('pytensor.operations'), inspect.isclass):
    setattr(Tensor, '_' + name.lower(), cls)

for name in ['add', 'sub', 'mul', 'pow', 'truediv']:
    fxn = getattr(Tensor, name)
    setattr(Tensor, f"__i{name}__", lambda self, x: self.assign(fxn(self.data, x)))
    setattr(Tensor, f"__r{name}__", lambda self, x: fxn(self.data, x))
