#!/usr/bin/env python3
from pytensor.tensor import Tensor
from pytensor.layers import Embedding, RNNCell, CrossEntropyLoss
from pytensor.optimizers import SGD
from datasets import single_sup_fact


class NN:
    def __init__(self):
        self.train_data, self.test_data, self.vocab = single_sup_fact.get_data()
        self.train_data = Tensor.np.array(self.train_data)
        self.test_data = Tensor.np.array(self.test_data)
        vocab_size = len(self.vocab)

        self.embed = Embedding(vocab_size, 16)
        self.model = RNNCell(16, 16, vocab_size)

        self.optimizer = SGD(self.model.params + self.embed.params)
        self.criterion = CrossEntropyLoss()

    def prediction(self, data, batch_size):
        hidden = self.model.init_hidden(batch_size)
        for t in range(5):
            input = Tensor(data[0:batch_size, t], autograd=True)
            rnn_input = self.embed.forward(input)
            output, hidden = self.model.forward(rnn_input, hidden)
        return output, hidden

    def train(self, epochs=1000, batch_size=10, lr=0.05, output=True):
        self.optimizer.lr = lr
        left = 0

        for i in range(epochs):
            total_loss = 0

            hidden = self.model.init_hidden(batch_size)
            for t in range(5):
                input = Tensor(self.train_data[left:batch_size, t], autograd=True)
                rnn_input = self.embed.forward(input)
                output, hidden = self.model.forward(rnn_input, hidden)

            target = Tensor(self.train_data[left:batch_size, t+1], autograd=True)
            loss = self.criterion.forward(output, target)
            loss.backward(Tensor.ones_like(loss.data))
            self.optimizer.step()
            total_loss += loss.data

            if output and (i % (epochs // 5) == 0):
                print(
                    f'Epoch: {i}.'
                    f'Loss: {total_loss / (len(self.train_data)/batch_size)}.'
                    f'Correct: {(target.data == Tensor.np.argmax(output.data, axis=1)).mean()}.'
                )

    def test(self):
        batch_size = 1
        hidden = self.model.init_hidden(batch_size)
        for t in range(5):
            input = Tensor(self.train_data[0:batch_size, t], autograd=True)
            rnn_input = self.embed.forward(input)
            output, hidden = self.model.forward(rnn_input, hidden)

        ctx = ''
        for idx in self.train_data[0:batch_size][0][0:-1]:
            ctx += self.vocab[idx] + ' '

        print(f'Context: {ctx}')
        print(f'Pred: {self.vocab[output.data.argmax()]}')


if __name__ == '__main__':
    nn = NN()
    nn.train()
    nn.test()
