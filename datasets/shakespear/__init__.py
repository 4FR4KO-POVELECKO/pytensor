def get_data(tiny=True):
    path = 'datasets/shakespear/shakespear_tiny.txt' if tiny else 'datasets/shakespear/shakespear.txt'
    f = open(path)
    raw = f.read()
    f.close()

    vocab = list(set(raw))
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i
    indices = list(map(lambda x: word2index[x], raw))

    return indices, vocab, word2index
