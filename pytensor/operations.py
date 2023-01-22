from pytensor.tensor import Operation, Tensor


class Add(Operation):
    def forward(self, x, y):
        return self.new_tensor(x + y)

    def backward(self, grad):
        self.parents[0].backward(grad)
        self.parents[1].backward(grad)


class Sub(Operation):
    def forward(self, x, y):
        return self.new_tensor(x - y)

    def backward(self, grad):
        new = self.new_tensor(grad.data)
        self.parents[0].backward(new)
        new = self.new_tensor(grad.neg().data)
        self.parents[1].backward(new)


class Mul(Operation):
    def forward(self, x, y):
        return self.new_tensor(x * y)

    def backward(self, grad):
        new = grad * self.parents[1]
        self.parents[0].backward(new)
        new = grad * self.parents[0]
        self.parents[1].backward(new)


class Pow(Operation):
    def forward(self, x, y):
        return self.new_tensor(x**y)

    def backward(self, grad):
        x = self.parents[0]
        y = self.parents[1]
        new = grad * y * (pow(x, y) / x)
        x.backward(new)
        new = grad * Tensor.log(x.data) * pow(x, y)
        y.backward(new)


class Truediv(Operation):
    def forward(self, x, y):
        return self.new_tensor(x / y)

    def backward(self, grad):
        new = grad / self.parents[1]
        self.parents[0].backward(new)
        new = -self.parents[0] / self.parents[1]**2
        self.parents[1].backward(new)


class Neg(Operation):
    def forward(self, x):
        return self.new_tensor(x * -1)

    def backward(self, grad):
        self.parents[0].backward(grad.neg())


class MatMul(Operation):
    def forward(self, x, y):
        return self.new_tensor(x.dot(y))

    def backward(self, grad):
        act = self.parents[0]
        weights = self.parents[1]
        new = grad.matmul(weights.transpose())
        act.backward(new)
        new = grad.transpose().matmul(act).transpose()
        weights.backward(new)


class Transpose(Operation):
    def forward(self, x):
        return self.new_tensor(x.transpose())

    def backward(self, grad):
        self.parents[0].backward(grad.transpose())


class Sum(Operation):
    def forward(self, x, dim):
        self.dim = dim
        return self.new_tensor(x.sum(dim))

    def backward(self, grad):
        ds = self.parents[0].data.shape[self.dim]
        self.parents[0].backward(grad.expand(self.dim, ds))


class Expand(Operation):
    def forward(self, x, dim, copies):
        self.dim = dim
        data_shape = x.shape
        trans_cmd = list(range(0, len(data_shape)))
        trans_cmd.insert(dim, len(data_shape))
        new_shape = list(data_shape) + [copies]
        new_data = x.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)
        return self.new_tensor(new_data)

    def backward(self, grad):
        self.parents[0].backward(grad.sum(self.dim))


class Sigmoid(Operation):
    def forward(self, x):
        new_data = 1 / (1 + Tensor.exp(-x))
        return self.new_tensor(new_data)

    def backward(self, grad):
        new = grad * (self.new * (Tensor.ones_like(grad) - self.new))
        self.parents[0].backward(new)
