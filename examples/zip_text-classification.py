import gzip
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict


class TextClassification:
    def __init__(self):
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space'
        ]

        train_df = fetch_20newsgroups(subset='train', categories=categories)
        self.test_df = fetch_20newsgroups(subset='test', categories=categories)

        self.label_texts = defaultdict(str)
        for i, text in enumerate(train_df['data']):
            label = train_df['target_names'][train_df['target'][i]]
            self.label_texts[label] += ' ' + text.lower()

        self.original_sizes = {
            label: len(gzip.compress(text.encode()))
            for label, text in self.label_texts.items()
        }

    def predict(self, text):
        sizes = {
            label: len(gzip.compress(f'{label_text} {text.lower()}'.encode()))
            for label, label_text in self.label_texts.items()
        }
        return min(sizes, key=lambda label: sizes[label] - self.original_sizes[label])
    
    def test(self):
        predictions = []
        for text in self.test_df['data']:
            predictions.append(self.predict(text))
        
        test_labels = [
            self.test_df['target_names'][label]
            for label in self.test_df['target']
        ]
        true_pred = 0
        for i, label in enumerate(test_labels):
            if label == predictions[i]:
                true_pred += 1
        print(f'T: {true_pred / (len(test_labels)/100)}')


if __name__ == '__main__':
    nn = TextClassification()
    nn.test()
