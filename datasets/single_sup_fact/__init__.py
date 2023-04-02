import re


def get_data():
    f = open('datasets/single_sup_fact/train.txt')
    train = f.readlines()
    f.close()

    f = open('datasets/single_sup_fact/test.txt')
    test = f.readlines()
    f.close()

    tokens, vocab = get_tokens_and_vocab(train)
    word2index = vocab2index_dict(vocab)

    train_data = list()
    for token in tokens:
        idx = words2indices(token, word2index)
        train_data.append(idx)

    tokens, _ = get_tokens_and_vocab(test)
    test_data = list()
    for token in tokens:
        idx = words2indices(token, word2index)
        test_data.append(idx)

    return train_data, test_data, list(vocab)


def get_tokens_and_vocab(data):
    tokens = list()
    vocab = set()
    for line in data:
        line = line.lower()
        line = re.sub(r'\d', '', line)
        line = line.replace('\n', '').replace('\t', '')
        line = line.replace('?', '').replace('.', '')
        line = line.split(' ')[1:]
        line = ['-'] * (6 - len(line)) + line
        tokens.append(line)
        for w in line:
            vocab.add(w)

    return tokens, vocab


def vocab2index_dict(vocab):
    word2index = {}
    for i, w in enumerate(vocab):
        word2index[w] = i
    return word2index


def words2indices(sentence, word_idx):
    idx = list()
    for w in sentence:
        idx.append(word_idx[w])
    return idx
