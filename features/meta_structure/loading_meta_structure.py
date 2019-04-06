import warnings, ast
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from datetime import datetime
from dateutil import parser

# config
np.random.seed(0)


class favCount_score(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_twitter = MinMaxScaler(feature_range=(0, 1), copy=False)
        self.scaler_reddit = MinMaxScaler(feature_range=(0, 1), copy=False)

    def get_value(self, x):
        x = ast.literal_eval(x)
        if 'favorite_count' in x:
            return float(x['favorite_count'])
        else:
            try:
                return float(x['score'])
            except:
                return float(0)

    def fit(self, X, y=None):
        X['content_fav_score'] = X['content'].map(lambda x: self.get_value(x))
        if len(X[X['dataset_source'].str.contains('twitter')]['content_fav_score'].values) > 0:
            self.scaler_twitter = self.scaler_twitter.fit(X[X['dataset_source'].str.contains('twitter')]['content_fav_score'].values.reshape((-1,1)))

        if len(X[X['dataset_source'].str.contains('reddit')]['content_fav_score'].values) > 0:
            self.scaler_reddit = self.scaler_reddit.fit(X[X['dataset_source'].str.contains('reddit')]['content_fav_score'].values.reshape((-1,1)))
        return self

    def transform(self, X):
        X = X.reset_index(drop=True)
        X['content_fav_score'] = X['content'].map(lambda x: self.get_value(x))

        for i, row in X.iterrows():
            if str(row['dataset_source']).__contains__('twitter'):
                twitter_val = self.scaler_twitter.transform(np.array(row['content_fav_score']).reshape(-1,1))[0][0]
                X.set_value(i, 'content_fav_score', twitter_val)
            else:
                reddit_val = self.scaler_reddit.transform(np.array(row['content_fav_score']).reshape(-1,1))[0][0]
                X.set_value(i, 'content_fav_score', reddit_val)
        return [[x] for x in X['content_fav_score'].tolist()]

class created_time(BaseEstimator, TransformerMixin):

    def get_time(self, x):
        x = ast.literal_eval(x)
        time_str = ""
        if 'created_at' in x:
            time_str = str(x['created_at'])
            full_date = parser.parse(time_str).replace(tzinfo=None)
            time_str = (full_date - datetime(1970, 1, 1)).total_seconds()
        else:
            try:
                time_str = str(x['created'])
            except:
                time_str = np.NaN

        return time_str

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        source_time = 0
        parent_time = 0
        source_not_existed = 0
        parent_not_existed = 0

        X = X.reset_index(drop=True)
        X['created_time'] = X['content'].map(lambda x: self.get_time(x))

        for i, row in X.iterrows():
            source = X[(X['event'] == row['event']) & (X['id'] == row['source_id'])]
            if len(source) > 0:
                source_time = source['created_time'].tolist()[0]
            else:
                source_not_existed += 1
                # source_time = row['created_time']
                # raise ValueError('Source   in  created_time  in meta_structure')

            parent = X[(X['event'] == row['event']) & (X['id'] == row['parent'])]
            if len(parent) > 0:
                parent_time = parent['created_time'].tolist()[0]
            elif row['parent'] == 'source':
                parent_time = source_time
            else:
                parent_time = source_time
                parent_not_existed += 1
                # print('#{}#'.format(row['event']), '   #{}#'.format(row['parent']))
                # raise ValueError('Parent   in  created_time  in meta_structure')
            feats.append([source_time, parent_time])
        if source_not_existed > 0 or parent_not_existed > 0:
            print('Missed values in the dataset: sources={},  parents={}'.format(source_not_existed, parent_not_existed))
        return [x for x in feats]

class replies_count(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = X['childs'].values
        feats = feats.reshape(-1, 1)

        return feats

class level(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = X['level'].values
        feats = feats.reshape(-1, 1)
        return feats


def meta_structure():
    feats = FeatureUnion([
            ('favCount_score', Pipeline([
                ('favCount_score', favCount_score()),
            ])),
            ('created_time', Pipeline([
                ('created_time', created_time()),
            ])),
            ('replies_count', Pipeline([
                ('replies_count', replies_count()),
            ])),
            ('level', Pipeline([
                ('level', level()),
            ])),
        ])
    return feats
