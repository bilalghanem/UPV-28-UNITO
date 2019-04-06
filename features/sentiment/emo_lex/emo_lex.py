import pandas as pd
from os.path import join


class emo_lex_class:
    def __init__(self, path=''):
        self.nrc = pd.read_csv(join(path, 'nrc.txt'), sep='\t', names=["word", "emotion", "association"])
        self.nrc = self.nrc.pivot(index='word', columns='emotion', values='association').reset_index()

    def score(self, sentence):
        words = sentence.split()
        pos = len(set(words).intersection(set(self.nrc[self.nrc['positive'] == 1]['word'].tolist())))
        neg = len(set(words).intersection(set(self.nrc[self.nrc['negative'] == 1]['word'].tolist())))
        return [pos, neg]

if __name__ == '__main__':
    s = ['love', 'hate', 'shit', 'hit']
    h = emo_lex_class()
    print(h.score('we are beautiful but bad'))
