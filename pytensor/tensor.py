import inspect
import importlib
import numpy as np


class Tensor(object):
    def __init__(self, data: list, autograd: bool = False):
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
    def randn(cls, *shape, **kwargs):
        return cls(np.random.rand(shape), **kwargs)

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def log(x):
        return np.log(x)

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
        tensors = [tensor for tensor in data if type(tensor) is Tensor]  # отбираем только тензоры
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

    def __setitem__(self, x, value): self.data[x] = value
    def __getitem__(self, x): return self.data[x]


class Operation(object):
    def __init__(self, *tensors: Tensor):
        self.parents = tensors
        self.autograd = any([tensor.autograd for tensor in tensors])

    def forward(self, *args):
        raise NotImplementedError(f'Метод forward не применим для {type(self)}')

    def backward(self, grad: Tensor):
        raise NotImplementedError(f'Метод backward не применим для {type(self)}')

    def new_tensor(self, data):
        tensor = Tensor(data, autograd=self.autograd)
        tensor.op = self
        return tensor


# Задаем операции
# operation.Add -> Tensor._add ...
for name, cls in inspect.getmembers(importlib.import_module('pytensor.operations'), inspect.isclass):
    setattr(Tensor, '_' + name.lower(), cls)

# Задаем операции с присваиванием и отраженные операции
for name in ['add', 'sub', 'mul', 'pow', 'truediv']:
    fxn = getattr(Tensor, name)
    setattr(Tensor, f"__i{name}__", lambda self, x: self.assign(fxn(self.data, x)))
    setattr(Tensor, f"__r{name}__", lambda self, x: fxn(self.data, x))
