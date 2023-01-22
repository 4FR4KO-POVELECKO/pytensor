from pytensor.tensor import Tensor


class Layer(object):
    def __init__(self):
        self.params = list()


class Linear(Layer):
    def __init__(self, in_size: int, out_size: int, bias: bool = True):
        """
        :param in_size(int): размер входной выборки
        :param out_size(int): размер выходной выборки
        :param bias(bool): если установлено значение False, слой будет обучаться без смещения. По умолчанию: True
        """
        super().__init__()
        self.weights = Tensor.randn(*(in_size, out_size), autograd=True)
        self.params.append(self.weights)
        self.bias = None
        if bias:
            self.bias = Tensor.zeros(out_size, autograd=True)
            self.params.append(self.bias)

    def forward(self, input_data):
        if self.bias:
            return input_data.matmul(self.weights) + self.bias.expand(0, len(input_data))
        return input_data.matmul(self.weights)


class Sequential(Layer):
    def __init__(self, layers: list = list()):
        super().__init__()
        self.layers = layers

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input_data):
        data = input_data
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def get_params(self):
        if not self.params:
            for layer in self.layers:
                self.params += layer.params
        return self.params


class MSELoss(Layer):
    def forward(self, prediction, target):
        return ((prediction - target)*(prediction - target)).sum(0)
