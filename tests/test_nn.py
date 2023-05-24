from examples.first import FirstNN


class TestNN:
    def test_first(self):
        nn = FirstNN()
        nn.train(output=False)
        assert nn.test() is True
