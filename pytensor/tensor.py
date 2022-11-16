import inspect
import importlib
import numpy as np


class Tensor(object):
    def __init__(self, data: list, autograd: bool = False):
        self.data = np.array(data)
        self.autograd = autograd
        self.grad: Tensor = None
        self.op: Operation = None

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
    def neg(self): return self.__neg__()

    def __add__(self, x):
        return self._execute(Tensor._add, self, x)

    def __neg__(self):
        return self._execute(Tensor._neg, self)

    def __sub__(self, x):
        return self._execute(Tensor._sub, self, x)

    def __mul__(self, x):
        return self._execute(Tensor._mul, self, x)

    def __repr__(self): return str(self.data.__repr__())
    def __str__(self): return str(self.data.__str__())


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
for name, cls in inspect.getmembers(importlib.import_module('pytensor.operation'), inspect.isclass):
    setattr(Tensor, '_' + name.lower(), cls)
