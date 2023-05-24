from pytensor.tensor import Tensor


class Layer(object):
    def __init__(self):
        self.params = list()


class Linear(Layer):
    def __init__(self, in_size: int, out_size: int, bias: bool = True):
        """
        :param in_size(int): input size
        :param out_size(int): output size
        :param bias(bool): default True
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


class Embedding(Layer):
    def __init__(self, voc_size: int, dim_size: int):
        """
        :param voc_size(int): vocab size
        :param dim_size(int): dimension size
        """
        super().__init__()
        w = Tensor.randn(*(voc_size, dim_size)) - 0.5 / dim_size
        self.weights = Tensor(w, autograd=True)
        self.params.append(self.weights)

    def forward(self, input_data):
        output = self.weights.getitem(input_data)
        return output


class RNNCell(Layer):
    def __init__(self, in_size: int, h_size: int, out_size: int, activation='sigmoid'):
        """
        :param in_size(int): input size
        :param h_size(int): hidden size
        :param out_size(int): output size
        :param activation(str, Activation, Activation()): activation class, default 'sigmoid'
        """
        super().__init__()
        self.hidden_size = h_size

        if isinstance(activation, str):
            activation = ACTIVATION_CLASSES.get(activation, None)
        if isinstance(activation, type):
            self.activation = activation()
        elif isinstance(activation, Activation):
            self.activation = activation
        else:
            self.activation = None

        if not self.activation:
            raise Exception(f'Non-linearity not found: {activation}.')

        self.w_ih = Linear(in_size, h_size)
        self.w_hh = Linear(h_size, h_size)
        self.w_ho = Linear(h_size, out_size)

        self.params += self.w_ih.params
        self.params += self.w_hh.params
        self.params += self.w_ho.params

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor.zeros(*(batch_size, self.hidden_size), autograd=True)


class LSTMCell(Layer):
    def __init__(self, in_size: int, h_size: int, out_size: int):
        """
        :param in_size(int): input size
        :param h_size(int): hidden size
        :param out_size(int): output size
        """
        super().__init__()
        self.hidden_size = h_size

        self.xf = Linear(in_size, h_size)
        self.xi = Linear(in_size, h_size)
        self.xo = Linear(in_size, h_size)
        self.xc = Linear(in_size, h_size)
        self.hf = Linear(h_size, h_size, bias=False)
        self.hi = Linear(h_size, h_size, bias=False)
        self.ho = Linear(h_size, h_size, bias=False)
        self.hc = Linear(h_size, h_size, bias=False)

        self.w_ho = Linear(h_size, out_size, bias=False)

        self.params += self.xf.params
        self.params += self.xi.params
        self.params += self.xo.params
        self.params += self.xc.params
        self.params += self.hf.params
        self.params += self.hi.params
        self.params += self.ho.params
        self.params += self.hc.params

        self.params += self.w_ho.params

    def forward(self, input, hidden):
        prev_hidden = hidden[0]
        prev_cell = hidden[1]

        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()
        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()
        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()
        g = (self.xc.forward(input) + self.hc.forward(prev_hidden)).tanh()
        c = (f * prev_cell) + (i * g)
        h = o * c.tanh()

        output = self.w_ho.forward(h)
        return output, (h, c)

    def init_hidden(self, batch_size=1):
        h = Tensor.zeros(*(batch_size, self.hidden_size), autograd=True)
        c = Tensor.zeros(*(batch_size, self.hidden_size), autograd=True)
        h.data[:, 0] += 1
        c.data[:, 0] += 1
        return (h, c)


class Sequential(Layer):
    def __init__(self, layers: list = list()):
        """
        :param layers(Layer): layers list
        """
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


# Loss

class MSELoss(Layer):
    def forward(self, prediction, target):
        return ((prediction - target)*(prediction - target)).sum(0)


class CrossEntropyLoss(Layer):
    def forward(self, prediction, target):
        return prediction.cross_entropy(target)


# Activation

class Activation:
    pass


class Sigmoid(Layer, Activation):
    def forward(self, input_data):
        return input_data.sigmoid()


class Tanh(Layer, Activation):
    def forward(self, input_data):
        return input_data.tanh()


ACTIVATION_CLASSES = {
    'sigmoid': Sigmoid,
    'tanh': Tanh,
}
