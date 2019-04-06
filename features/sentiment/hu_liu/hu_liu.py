import pandas as pd



class hu_liu_class:
    def __init__(self, path=''):
        pos = path + 'positive-words.txt'
        neg = path + 'negative-words.txt'
        self.pos = pd.read_csv(pos, skiprows=35, names=['words'], encoding='utf-8')
        self.neg = pd.read_csv(neg, skiprows=35, names=['words'], encoding='latin1')

    def score(self, sentence):
        words = sentence.split()
        pos = len(set(words).intersection(set(self.pos['words'].tolist())))# / len(words)
        neg = len(set(words).intersection(set(self.neg['words'].tolist())))# / len(words)

        return [pos, neg]

if __name__ == '__main__':
    s = ['love', 'hate', 'shit', 'hit']
    h = hu_liu_class()
    print(h.score('we are beautiful but mal'))
