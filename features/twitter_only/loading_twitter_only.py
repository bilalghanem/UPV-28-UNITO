import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re, ast


# config
np.random.seed(0)


class mention_tag(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def counting(self, text):
        mention = len(re.findall(r'usermention', text))
        tags = len(re.findall(r'hashtag', text))
        return [mention, tags]

    def transform(self, X):
        feats = []
        for i, row in X.iterrows():
            if str(row['dataset_source']).__contains__('twitter'):
                feats.append(self.counting(row['text']))
            else:
                feats.append([np.NaN, np.NaN])
        return [x for x in feats]

class retweet_count(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for i, row in X.iterrows():
            if str(row['dataset_source']).__contains__('twitter'):
                content = ast.literal_eval(row['content'])
                feats.append([content['retweet_count']])
            else:
                feats.append([np.NaN])
        return [x for x in feats]

class user_feats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for i, row in X.iterrows():
            if str(row['dataset_source']).__contains__('twitter'):
                content = ast.literal_eval(row['content'])
                feats.append([1 if content['user']['verified'] else 0, content['user']['followers_count'], content['user']['listed_count'], content['user']['statuses_count'], content['user']['friends_count'],
                              content['user']['favourites_count']])
            else:
                feats.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])

        return [x for x in feats]


def twitter_feats():
    feats = FeatureUnion([
        ('mention_tag_url', Pipeline([
            ('mention_tag_url', mention_tag()),
        ])),
        ('retweet_count', Pipeline([
            ('retweet_count', retweet_count()),
        ])),
        ('user_feats', Pipeline([
            ('user_feats', user_feats()),
        ]))
    ])
    return feats

def one_vector_twitter_feats(sentences):
        res = twitter_feats().fit_transform(sentences)
        return res
