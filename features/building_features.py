import warnings, re
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer

import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
from os.path import join
from joblib import Parallel, delayed
from features.emotional.loading_emotional_lexicons import emotional_lexicons
from features.sentiment.loading_sentiment_analyzer import Sentiment
from features.lexical.loading_lexical import Lexical
from features.syntactical.loading_syntactic import Syntactic
from features.stylistic.loading_stylistic import one_vector_stylistic
from features.meta_structure.loading_meta_structure import meta_structure
from features.twitter_only.loading_twitter_only import one_vector_twitter_feats

# config
np.random.seed(0)

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X[[self.column]][self.column]

class sep_special_chars(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.map(lambda x: re.sub(r'([?¿!¡.])', r' \1 ', x))
        return X


class emotional_features(BaseEstimator, TransformerMixin):
    def __init__(self, aggregated=False, binary=False, path='', data_type='train'):
        self.aggregated = aggregated
        self.binary = binary
        self.path = path
        self.train_len = 0
    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/emotional_features_{}.npy'.format(data_type))
        if exists(file_name):
            features = np.load(file_name)
        else:
            emo = emotional_lexicons(path=join(self.path, 'emotional'))
            if self.aggregated:
                loop = tqdm(X)
                loop.set_description('Building emotional_features ({})'.format(data_type))
                features = [emo.aggregated_vector_emo(sentence, self.binary) for sentence in loop]
                features = np.nan_to_num(np.array(features))
                np.save(file_name, features)
            else:
                loop = tqdm(X)
                loop.set_description('Building emotional_features ({})'.format(data_type))
                # features = [emo.one_vector_emo(sentence) for sentence in loop]
                features = Parallel(n_jobs=-1)(delayed(emo.one_vector_emo)(sentence) for sentence in loop)
                features = np.nan_to_num(np.array(features))
                np.save(file_name, features)

        return [x for x in features]

class sentiment_features(BaseEstimator, TransformerMixin):
    def __init__(self, path='', data_type='train'):
        self.path = path
        self.train_len = 0

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/sentiment_features_{}.npy'.format(data_type))
        if exists(file_name):
            features = np.load(file_name)
        else:
            senti = Sentiment(path=join(self.path, 'sentiment'))
            loop = tqdm(X)
            loop.set_description('Building Sentiment_features ({})'.format(data_type))
            # features = [senti.one_vector_senti(sentence) for sentence in loop]
            features = Parallel(n_jobs=-1)(delayed(senti.one_vector_senti)(sentence) for sentence in loop)
            features = np.nan_to_num(np.array(features))
            np.save(file_name, features)
        return [x for x in features]

class lexical_features(BaseEstimator, TransformerMixin):
    def __init__(self, path=''):
        self.path = path
        self.train_len = 0

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/lexical_features_{}.npy'.format(data_type))
        if exists(file_name):
            features = np.load(file_name)
        else:
            lex = Lexical(path=join(self.path, 'lexical'))
            loop = tqdm(X)
            loop.set_description('Building Lexical_features ({})'.format(data_type))
            # features = [lex.one_vector_lexical(sentence) for sentence in loop]
            features = Parallel(n_jobs=-1)(delayed(lex.one_vector_lexical)(sentence) for sentence in loop)
            features = np.nan_to_num(np.array(features))
            np.save(file_name, features)
        return [x for x in features]

class syntactic_features(BaseEstimator, TransformerMixin):
    def __init__(self, path='', udpipe=False):
        self.udpipe = udpipe
        self.path = path
        self.train_len = 0

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/syntactic_features_{}.npy'.format(data_type))
        if exists(file_name):
            features = np.load(file_name)
        else:
            syn = Syntactic(udpipe=self.udpipe)
            loop = tqdm(X)
            loop.set_description('Building Syntactic_features ({})'.format(data_type))
            # features = [syn.one_vector_synt(sentence) for sentence in loop]
            features = Parallel(n_jobs=1)(delayed(syn.one_vector_synt)(sentence) for sentence in loop)
            features = np.nan_to_num(np.array(features))
            np.save(file_name, features)
        return [x for x in features]

class stylistic_features(BaseEstimator, TransformerMixin):
    def __init__(self, path=''):
        self.path = path
        self.train_len = 0

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/stylistic_features_{}.npy'.format(data_type))
        if exists(file_name):
            features = np.load(file_name)
        else:
            loop = tqdm(X)
            loop.set_description('Building Stylistic_features ({})'.format(data_type))
            features = one_vector_stylistic(X)
            loop.update(len(X))
            features = np.nan_to_num(np.array(features))
            np.save(file_name, features)
        return [x for x in features]

class meta_structure_features(BaseEstimator, TransformerMixin):
    def __init__(self, path=''):
        self.path = path
        self.train_len = 0
        self.features = meta_structure()

    def fit(self, X, y=None):
        self.train_len = len(X)
        self.features = self.features.fit(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        else:
            data_type = 'test'

        file_name = join(self.path, 'preprocessed_data/meta_structure_features_{}.npy'.format(data_type))
        if exists(file_name):
            features = np.load(file_name)
        else:
            loop = tqdm(X)
            loop.set_description('Building Meta & Structural features ({})'.format(data_type))
            X = self.features.transform(X)
            loop.update(len(X))
            features = np.nan_to_num(np.array(X))
            np.save(file_name, features)
        return [x for x in features]

class twitter_only_features(BaseEstimator, TransformerMixin):
    def __init__(self, path=''):
        self.path = path
        self.train_len = 0

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        else:
            data_type = 'test'

        file_name = join(self.path, 'preprocessed_data/twitter_only_features_{}.npy'.format(data_type))
        if exists(file_name):
            features = np.load(file_name)
        else:
            loop = tqdm(X)
            loop.set_description('Building Twitter Only features ({})'.format(data_type))
            X = one_vector_twitter_feats(X)
            loop.update(len(X))
            features = np.nan_to_num(np.array(X))
            np.save(file_name, features)
        return [x for x in features]



def get_manual_features(path='', n_jobs=1):
    manual_feats = FeatureUnion([
            ('emotional_features_pip', Pipeline([ #--B
                ('column', ColumnSelector('text')),
                ('sep_speciall_chars', sep_special_chars()),
                ('emotional_features', emotional_features(aggregated=False, path=path)),
            ])),
            ('sentiment_features_pip', Pipeline([ #--C
                ('column', ColumnSelector('text')),
                ('sep_speciall_chars', sep_special_chars()),
                ('sentiment_features', sentiment_features(path=path)),
            ])),
            ('lexical_features_pip', Pipeline([ #--D
                ('column', ColumnSelector('text')),
                ('sep_speciall_chars', sep_special_chars()),
                ('lexical_features', lexical_features(path=path)),
            ])),
            ('syntactic_features_pip', Pipeline([ #--E
                ('column', ColumnSelector('text')),
                ('sep_special_chars', sep_special_chars()),
                ('syntactic_features', syntactic_features(path=path, udpipe=True)),
            ])),
            ('stylistic_features_pip', Pipeline([ #--F
                ('column', ColumnSelector('text')),
                ('stylistic_features', stylistic_features(path=path)),
            ])),
            ('meta_structure_features_pip', Pipeline([ #--G
                ('meta_structure_features', meta_structure_features(path=path)),
            ])),
            ('twitter_only_features_pip', Pipeline([ #--H
                ('twitter_only_features', twitter_only_features(path=path)),
                ('Imputer', Imputer(missing_values=np.NaN, strategy='mean', axis=0))
            ]))
        ], n_jobs=n_jobs)

    return manual_feats



if __name__ == '__main__':
    s = ["I don't want to be sad"]
    res = get_manual_features().fit_transform(s)
    print(res)
