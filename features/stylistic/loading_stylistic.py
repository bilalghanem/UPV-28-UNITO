import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import groupby
import numpy as np
import re

# config
np.random.seed(0)


class count_question(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def question_marks(self, text):
        text = re.sub(r'([^a-zA-Z0-9])',r' \1 ', text)
        text = len(re.findall(r'[?¿]', text))
        return text

    def transform(self, X):
        feats = []
        for sent in X:
            feats.append(self.question_marks(sent))
        return [[x] for x in feats]

class count_exclamation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def exclamation_marks(self, text):
        text = re.sub(r'([^a-zA-Z0-9])',r' \1 ', text)
        text = len(re.findall(r'[!¡]', text))
        return text

    def transform(self, X):
        feats = []
        for sent in X:
            feats.append(self.exclamation_marks(sent))
        return [[x] for x in feats]

class sentence_length(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for sent in X:
            feats.append(len(sent.split()))
        return [[x] for x in feats]

class count_consecutive(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def consecutive_chars(self, text):
        count = sum([count for count in [sum(1 for _ in group) for label, group in groupby(text)] if count > 1])
        return count

    def transform(self, X):
        length_list=[]
        for sent in X:
            length_list.append(self.consecutive_chars(sent))
        return [[x] for x in length_list]

class uppercase_ratio(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def upper_case(self, text):
        up_count = len(re.findall(r'[A-Z]', text))
        low_count = len(re.findall(r'[a-z]', text))
        try:
            res = (up_count/low_count)
        except:
            res = 1
        return res

    def transform(self, X):
        feats = []
        for sent in X:
            feats.append(self.upper_case(sent))
        return [[x] for x in feats]

class url(BaseEstimator, TransformerMixin): ############# check the hashtags mentions
    def fit(self, X, y=None):
        return self

    def counting(self, text):
        url = len(re.findall(r'http', text))
        return [url]

    def transform(self, X):
        feats = []
        for sent in X:
            feats.append(self.counting(sent))
        return [x for x in feats]


def stylistic():
    feats = FeatureUnion([
            ('count_question', Pipeline([
                ('count_question', count_question()),
            ])),
            ('count_exclamation', Pipeline([
                ('count_exclamation', count_exclamation()),
            ])),
            ('sentence_length', Pipeline([
                ('sentence_length', sentence_length()),
            ])),
            ('count_consecutive', Pipeline([
                ('count_consecutive', count_consecutive()),
            ])),
            ('uppercase_ratio', Pipeline([
                ('uppercase_ratio', uppercase_ratio()),
            ])),
            ('mention_tag_url', Pipeline([
                ('mention_tag_url', url()),
            ]))
        ])
    return feats

def one_vector_stylistic(sentences):
        res = stylistic().transform(sentences)
        return res
