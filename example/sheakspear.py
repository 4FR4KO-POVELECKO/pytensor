#!/usr/bin/env python3
from pytensor.tensor import Tensor
from pytensor.layers import Embedding, LSTMCell, CrossEntropyLoss
from pytensor.optimizers import SGD
from datasets.shakespear import get_data


class NN:
    # TODO: check the functionality

    def __init__(self):
        self.data, self.vocab, self.word2index = get_data()
        self.indices = Tensor.np.array(self.data)
        self.vocab = Tensor.np.array(self.vocab)
        self.embed = Embedding(vocab_size=len(self.vocab), dim=512)
        self.model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(self.vocab))
        self.model.w_ho.weights.data *= 0
        self.criterion = CrossEntropyLoss()
        self.optim = SGD(params=(self.model.params + self.embed.params))

    def generate_sample(self, n=30, init_char=' '):
        s = ''
        hidden = self.model.init_hidden(batch_size=1)
        e_input = Tensor(Tensor.np.array(self.word2index[init_char]))
        for i in range(n):
            rnn_input = self.embed.forward(e_input)
            output, hidden = self.model.forward(rnn_input, hidden)
            output.data *= 10
            temp_dist = output.softmax()
            temp_dist /= temp_dist.sum(0)

            m = (temp_dist.data > Tensor.np.random.rand()).argmax()
            c = self.vocab[m]
            e_input = Tensor(Tensor.np.array([m]))
            s += c

        return s

    def train(self, epochs=100, batch_size=16, bptt=25, lr=0.05, output=True):
        self.optim.lr = lr
        n_batches = int(self.indices.shape[0] / batch_size)
        trimmed_indices = self.indices[:n_batches*batch_size]
        batched_indices = trimmed_indices.reshape(batch_size, n_batches)
        batched_indices = batched_indices.transpose()

        input_batched_indices = batched_indices[0: -1]
        target_batched_indices = batched_indices[1:]

        n_bptt = int((n_batches-1) / bptt)

        input_batches = input_batched_indices[:n_bptt*bptt]
        input_batches = input_batches.reshape(n_bptt, bptt, batch_size)
        target_batches = target_batched_indices[:n_bptt*bptt]
        target_batches = target_batches.reshape(n_bptt, bptt, batch_size)

        for i in range(epochs):
            total_loss = 0

            hidden = self.model.init_hidden(batch_size=batch_size)
            for b in range(len(input_batches)):
                hidden = (Tensor(hidden[0].data, autograd=True),
                          Tensor(hidden[1].data, autograd=True))
                loss = None
                losses = list()
                for t in range(bptt):
                    e_input = Tensor(input_batches[b][t], autograd=True)
                    rnn_input = self.embed.forward(e_input)
                    output, hidden = self.model.forward(rnn_input, hidden)

                    target = Tensor(target_batches[b][t], autograd=True)
                    batch_loss = self.criterion.forward(output, target)
                    losses.append(batch_loss)
                    loss = batch_loss if t == 0 else loss + batch_loss

                for loss in losses:
                    loss.backward(Tensor.ones_like(loss.data))
                    self.optim.step()
                    total_loss += loss.data
                log = (
                    f'\r Iter: {i}'
                    f' - Batch: {b+1}/{len(input_batches)}'
                    f' - Loss: {total_loss / (i+1)}'
                )
                if b == 0:
                    log += ' - ' + self.generate_sample(70, '\n').replace('n', ' ')
                if b % 10 == 0 or b == len(input_batches):
                    print(log)
            self.optim.lr *= 0.99

    # TODO: def test(self):


if __name__ == '__main__':
    nn = NN()
    nn.train()
    # nn.test()
