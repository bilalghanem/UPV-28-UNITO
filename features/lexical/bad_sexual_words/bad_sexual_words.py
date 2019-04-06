import pandas as pd
from os.path import join

class bad_sexual_words_class:
    def __init__(self, path=''):

        self.bad_words = pd.read_excel(join(path, 'en_bad_words.xlsx'), names=['words'], encoding='utf-8')
        self.sexual_words = pd.read_excel(join(path, 'en_sexual_words.xlsx'), names=['words'], encoding='utf-8')
    
    def score(self, sentence):
        words = sentence.split()
        bad = len(set(words).intersection(set(self.bad_words['words'].tolist())))
        sexual = len(set(words).intersection(set(self.sexual_words['words'].tolist())))
        return [bad, sexual]

if __name__ == '__main__':
    sentence = 'fuck you'
    h = bad_sexual_words_class()
    print(h.score(sentence))
