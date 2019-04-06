import pandas as pd
from os.path import join

class cue_words_class:
    def __init__(self, path=''):
        self.words_belief = pd.read_csv(join(path, 'belief.txt'), names=['words'], encoding='utf-8')
        self.words_denial = pd.read_csv(join(path, 'denial.txt'), names=['words'], encoding='utf-8')
        self.words_doubt = pd.read_csv(join(path, 'doubt.txt'), names=['words'], encoding='utf-8')
        self.words_fake = pd.read_csv(join(path, 'fake.txt'), names=['words'], encoding='utf-8')
        self.words_knowledge = pd.read_csv(join(path, 'knowledge.txt'), names=['words'], encoding='utf-8')
        self.words_negation = pd.read_csv(join(path, 'negation.txt'), names=['words'], encoding='utf-8')
        self.words_question = pd.read_csv(join(path, 'question.txt'), names=['words'], encoding='utf-8')
        self.words_report = pd.read_csv(join(path, 'report.txt'), names=['words'], encoding='utf-8')
        print('')

    def score(self, sentence):
        words = sentence.split()
        words_belief = len(set(words).intersection(set(self.words_belief['words'].tolist())))
        words_denial = len(set(words).intersection(set(self.words_denial['words'].tolist())))
        words_doubt = len(set(words).intersection(set(self.words_doubt['words'].tolist())))
        words_fake = len(set(words).intersection(set(self.words_fake['words'].tolist())))
        words_knowledge = len(set(words).intersection(set(self.words_knowledge['words'].tolist())))
        words_negation = len(set(words).intersection(set(self.words_negation['words'].tolist())))
        words_question = len(set(words).intersection(set(self.words_question['words'].tolist())))
        words_report = len(set(words).intersection(set(self.words_report['words'].tolist())))
        return [words_belief, words_denial, words_doubt, words_fake, words_knowledge, words_negation, words_question, words_report]

if __name__ == '__main__':
    sentence = "I'm wondering, why there are not student here?"
    h = cue_words_class()
    print(h.score(sentence))
