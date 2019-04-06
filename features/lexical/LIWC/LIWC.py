from os.path import join
import pandas as pd
import re
from features.lexical.LIWC.liwc_readDict import readDict




class LIWC_class:
    def __init__(self, path=''):
        # LIWC, sad, anger, neg & pos emotion
        self.liwc = readDict(join(path, 'liwc.dic'))
        self.liwc = pd.DataFrame(self.liwc, columns=['word', 'category'])
        self.liwc['word'] = self.liwc['word'].map(lambda x: re.sub(r'[*]', '', x))
        self.liwc['value'] = 1
        self.liwc = pd.pivot_table(self.liwc, index='word', columns=['category'],
                                   values='value', fill_value=0).reset_index().reindex(['word', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'negate', 'swear', 'cause', 'certain', 'sexual'], axis=1)
        self.liwc.rename(columns={'negemo': 'negative_emotion', 'posemo': 'positive_emotion', 'sad': 'sadness'}, inplace=True)

    def score(self, sentence):
        words = sentence.split()

        i = len(set(words).intersection(set(self.liwc[self.liwc['i'] == 1]['word'].tolist())))
        we = len(set(words).intersection(set(self.liwc[self.liwc['we'] == 1]['word'].tolist())))
        you = len(set(words).intersection(set(self.liwc[self.liwc['you'] == 1]['word'].tolist())))
        shehe = len(set(words).intersection(set(self.liwc[self.liwc['shehe'] == 1]['word'].tolist())))
        they = len(set(words).intersection(set(self.liwc[self.liwc['they'] == 1]['word'].tolist())))
        ipron = len(set(words).intersection(set(self.liwc[self.liwc['ipron'] == 1]['word'].tolist())))
        negate = len(set(words).intersection(set(self.liwc[self.liwc['negate'] == 1]['word'].tolist())))
        swear = len(set(words).intersection(set(self.liwc[self.liwc['swear'] == 1]['word'].tolist())))
        cause = len(set(words).intersection(set(self.liwc[self.liwc['cause'] == 1]['word'].tolist())))
        certain = len(set(words).intersection(set(self.liwc[self.liwc['certain'] == 1]['word'].tolist())))
        sexual = len(set(words).intersection(set(self.liwc[self.liwc['sexual'] == 1]['word'].tolist())))
        return [i, we, you, shehe, they, ipron, negate, swear, cause, certain, sexual]

if __name__ == '__main__':
    sentence = "i'm we wondering, why there are not student here?"
    h = LIWC_class()
    print(h.score(sentence))
