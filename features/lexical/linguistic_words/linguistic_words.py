import pandas as pd
from os.path import join
import re



class linguistic_words_class:
    def __init__(self, path=''):
        self.assertives = pd.read_csv(join(path, 'assertives.txt'), names=['words'], encoding='utf-8')
        self.bias = pd.read_csv(join(path, 'bias.txt'), names=['words'], encoding='utf-8')
        self.factives = pd.read_csv(join(path, 'factives.txt'), names=['words'], encoding='utf-8')
        self.hedges = pd.read_csv(join(path, 'hedges.txt'), names=['words'], encoding='utf-8')
        self.implicatives = pd.read_csv(join(path, 'implicatives.txt'), names=['words'], encoding='utf-8')
        self.report_verbs = pd.read_csv(join(path, 'report_verbs.txt'), names=['words'], encoding='utf-8')

        self.subj = pd.read_csv(join(path, 'subjclueslen1-HLTEMNLP05.tff'), encoding='utf-8', sep=' ', names=['subj_type', '1', 'words', '3', '4', '5'])
        self.subj.drop(['1', '3', '4', '5'], axis=1, inplace=True)
        self.subj = self.subj.applymap(lambda x: re.findall('(?<=[a-z0-9]=)\w+', x)[0])
        print('')

    def score(self, sentence):
        words = sentence.split()
        assertives = len(set(words).intersection(set(self.assertives['words'].tolist())))
        bias = len(set(words).intersection(set(self.bias['words'].tolist())))
        factives = len(set(words).intersection(set(self.factives['words'].tolist())))
        hedges = len(set(words).intersection(set(self.hedges['words'].tolist())))
        implicatives = len(set(words).intersection(set(self.implicatives['words'].tolist())))
        report_verbs = len(set(words).intersection(set(self.report_verbs['words'].tolist())))
        strong_subj = len(set(words).intersection(set(self.subj[self.subj['subj_type'] == 'strongsubj']['words'].tolist())))
        weak_subj = len(set(words).intersection(set(self.subj[self.subj['subj_type'] == 'weaksubj']['words'].tolist())))
        return [assertives, bias, factives, hedges, implicatives, report_verbs, strong_subj, weak_subj]

if __name__ == '__main__':
    sentence = "I'm wondering, why there are not student here?"
    h = linguistic_words_class()
    print(h.score(sentence))
