import numpy as np


class Tensor(object):
    def __init__(self, data, autograd=False,
                 creators=None, op=None):
        self.data = np.array(data)
        self.op = op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.child = {}
        self._id = np.random.randint(0, 100000)
         
        if creators is not None:
            for c in creators:
                if self._id not in c.child:
                    c.child[self._id] = 1
                else:
                    c.child[self._id] += 1
    
    def check_child_grads(self):
        for _, cnt in self.child.items():
            if cnt != 0:
                return False
        return True
    
    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                if self.child[grad_origin._id] == 0:
                    raise Exception('Нельзя распространить больше одного раза.')
                else:
                    self.child[grad_origin._id] -= 1
            if self.grad:
                self.grad += grad
            else:
                self.grad = grad
                
            if self.creators and (self.check_child_grads() or not grad_origin):
                if self.op == 'add':
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                if self.op == 'neg':
                    self.creators[0].backward(self.grad.__neg__())
                if self.op == 'sub':
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)
                if self.op == 'mul':
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)
                if self.op == 'mmul':
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mmul(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mmul(act).transpose()
                    weights.backward(new)
                if self.op == 'transpose':
                    self.creators[0].backward(self.grad.transpose())
                if 'sum' in self.op:
                    dim = int(self.op.split('_')[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, ds))
                if 'expand' in self.op:
                    dim = int(self.op.split('_')[1])
                    self.creators[0].backward(self.grad.sum(dim))

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          op='sum_' + str(dim))
        return Tensor(self.data.sum(dim))
    
    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data, autograd=True,
                          creators=[self], op='expand_' + str(dim))
        return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), autograd=True,
                          creators=[self], op='transpose')
        return Tensor(self.data.transpose())

    def mmul(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data), autograd=True,
                          creators=[self, x], op='mmul')
        return Tensor(self.data.dot(x.data))
    
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          op='add')
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, autograd=True,
                          creators=[self], op='neg')
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          op='sub')
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          op='mul')
        return Tensor(self.data * other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
