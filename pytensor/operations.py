import numpy as np
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
        new = grad * np.log(x.data) * pow(x, y)
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
        ds = self.parents[0].shape[self.dim]
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
        new_data = 0.5 * (1 + np.tanh(0.5 * x))
        return self.new_tensor(new_data)

    def backward(self, grad):
        new = grad * (self.new * (Tensor.ones_like(grad) - self.new))
        self.parents[0].backward(new)


class Tanh(Operation):
    def forward(self, x):
        return self.new_tensor(np.tanh(x))

    def backward(self, grad):
        new = grad * (Tensor.ones_like(grad) - (self.new * self.new))
        self.parents[0].backward(new)


class Softmax(Operation):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.new_tensor(self.output)

    def backward(self, grad):
        jacobian = np.expand_dims(self.output, axis=-1) * \
                   np.expand_dims(np.eye(self.output.shape[-1]), axis=0)
        diagonal = np.expand_dims(self.output, axis=-1) * (1 - np.expand_dims(self.output, axis=-2))
        new = np.sum(jacobian - diagonal, axis=-2) @ grad
        self.parents[0].backward(new)


class GetItem(Operation):
    def forward(self, x, idx):
        new = self.new_tensor(x[idx])
        new.save_idx = idx if isinstance(idx, np.ndarray) else np.array(x)
        return new

    def backward(self, grad):
        new = Tensor.zeros_like(self.parents[0].data)
        indices = self.new.save_idx.flatten()
        grad = grad.data.reshape(len(indices), -1)
        for i in range(len(indices)):
            new[indices[i]] += grad[i]
        self.parents[0].backward(new)


class CrossEntropy(Operation):
    def forward(self, x, target):
        x_max = np.max(x.data, axis=len(x.shape)-1, keepdims=True)
        x_exp = np.exp(x.data - x_max)
        softmax_output = x_exp / np.sum(x_exp, axis=len(x.shape)-1, keepdims=True)
        t = target.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]

        eps = 1e-7
        loss = -(np.log(p + eps) * target_dist).sum(1).mean()

        new = self.new_tensor(loss)
        new.softmax_output = softmax_output
        new.target_dist = target_dist
        return new

    def backward(self, grad):
        dx = self.new.softmax_output - self.new.target_dist
        self.parents[0].backward(Tensor(dx))
