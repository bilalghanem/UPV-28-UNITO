import warnings

warnings.filterwarnings("ignore")
# Sklearn Package
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Files
from features.avg_consine_manual_feats import cosine_parent_source_feats, avg_parent_source_feats
from features.avg_consine_word2vec import cosine_parent_source_w2v, avg_parent_source_w2v
from features.building_features import get_manual_features

import re
import numpy as np
from os.path import join
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
stemmer = SnowballStemmer('english')

# config
np.random.seed(0)


class classification:

    def __init__(self, scoring='accuracy',  verbose=0, n_jobs=1):
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs

    def score(self, y_test, y_pred):
        if self.scoring == 'accuracy':
            print('Accuracy (Test): ', accuracy_score(y_test, y_pred))
        elif self.scoring.__contains__('f1'):
            print('Macro-F1 (Test): ', f1_score(y_test, y_pred, average='macro'))
        else:
            raise ValueError('The specificed score is not correct')

    def shuffle_numpy(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p, :], b.values[p]

    def run_clf(self, X_train, y_train, X_test, y_test, test_labels=True, apply_cv=False, main_path=''):
        path = join(main_path, 'features')

        full_model = Pipeline([
            ('PIP_ALL', Pipeline([
                ('contexual_feats', FeatureUnion([
                    ('Pip_manual', Pipeline([
                        ('clean_text', clean_data(remove_stopwords=False, column='text')),
                        ('manual_features', get_manual_features(path=path, n_jobs=self.n_jobs)),
                    ])),

                    ###================= Cosine
                    ('cosine_parent_source_feats', Pipeline([
                        ('cosine_parent_source_feats', cosine_parent_source_feats(path=path, n_jobs=self.n_jobs)),
                    ])),
                    ('cosine_parent_source_w2v', Pipeline([
                        ('clean_text', clean_data(remove_stopwords=True, column='text')),
                        ('cosine_parent_source_w2v', cosine_parent_source_w2v(path=path, n_jobs=self.n_jobs, embeddings_path='D:/GoogleNews-vectors-negative300.bin')),
                    ])),

                    ######================= AVG
                    ('avg_parent_source_feats', Pipeline([
                        ('avg_parent_source_feats', avg_parent_source_feats(path=path, n_jobs=self.n_jobs)),
                    ])),
                    ('avg_parent_source_w2v', Pipeline([
                        ('clean_text', clean_data(remove_stopwords=True, column='text')),
                        ('avg_parent_source_w2v', avg_parent_source_w2v(path=path, n_jobs=self.n_jobs, embeddings_path='D:/GoogleNews-vectors-negative300.bin')),
                    ])),
                ])),
                ('Normalization', StandardScaler()),
            ])),
            # ('classffier', LogisticRegression(n_jobs=-1, C=61.5848, penalty='l2', class_weight={'deny': 0.35, 'comment': 0.1, 'support': 0.2, 'query': 0.35})),
        ])

        if test_labels == True:
            if apply_cv:
                features = full_model.named_steps['PIP_ALL']
                X_train = features.fit_transform(X_train, y_train)
                X_train, y_train = self.shuffle_numpy(X_train, y_train)

                model_f1 = LogisticRegression(n_jobs=self.n_jobs, C=37.9269, penalty='l2', class_weight={'comment': 0.1, 'deny': 0.35, 'support': 0.2, 'query': 0.35})
                scores = cross_val_score(model_f1, X_train, y_train, cv=10, n_jobs=self.n_jobs, verbose=0, scoring='f1_macro')
                print('Mean CV {}: {},\tstd: {}\n'.format('f1_macro', round(float(np.mean(scores)), 3), round(float(np.std(scores)), 3)))

            else:
                clf = full_model.fit(X_train, y_train)
                predicted = clf.predict(X_test)
                self.score(y_test, predicted)
        else:
            X_test_BK = X_test
            X_test_BK['label'] = 0

            features = full_model.named_steps['PIP_ALL']
            X_train = features.fit_transform(X_train, y_train) #.append(X_test)
            X_test = features.transform(X_test)
            X_train, y_train = self.shuffle_numpy(X_train, y_train)
            model_f1 = LogisticRegression(n_jobs=self.n_jobs, C=37.9269, penalty='l2', class_weight={'comment': 0.1, 'deny': 0.35, 'support': 0.2, 'query': 0.35})

            scores = cross_val_score(model_f1, X_train, y_train, cv=10, n_jobs=self.n_jobs, verbose=0, scoring='f1_macro')
            print('Mean CV {}: {},\tstd: {}\n'.format('f1_macro', round(float(np.mean(scores)), 3), round(float(np.std(scores)), 3)))

            clf = model_f1.fit(X_train, y_train)
            predicted = clf.predict(X_test)
            X_test_BK['label'] = predicted
            subtaskaenglish = dict(zip(X_test_BK.id, X_test_BK.label))
            submission = {}
            submission['subtaskaenglish'] = subtaskaenglish
            submission['subtaskbenglish'] = {}

            # with open('answer.json', 'w') as outfile:
            #     json.dump(submission, outfile)
            # print('############### Saved to JSON')


class clean_data(BaseEstimator, TransformerMixin):
    def __init__(self, stemming=False, remove_stopwords=False, column=''):
        self.column = column
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords

    def clean_data(self, text):
        text = re.sub(r"((http|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=;%&:/~+#-]*[\w@?^=%&;:/~+#-])?)", " http ", text)
        text = re.sub(r'@(\w+)', ' usermention ', text) # @USER_MENTION
        text = re.sub(r'#(\w+)', ' hashtag ', text) # #HASH_TAG
        text = re.sub(r'[^a-zA-Z\'?¿!¡.]', ' ', text) # remove uneeded special characters
        text = re.sub(r"((?<=\s)'|'(?=\s)|^')", "", text) # ' that is not existed based on grammar

        text = re.sub(r"(?<=\w)'s", " ", text) # author's
        text = re.sub(r"isn't", "is not", text) # isn't
        text = re.sub(r"(?<=\w)'re", " are", text) # you're
        text = re.sub(r"(?<=\w)n't", " not", text) # you're

        text = re.sub(r'\s{2,}', ' ', text) # remove extra spaces

        if self.stemming:
            text = text.split()
            for i in range(len(text)):
                text[i] = stemmer.stem(text[i])
            text = ' '.join(text)
        if self.remove_stopwords:
                text = text.split()
                stops = set(stopwords.words("english"))
                stops.remove('not')
                text = [w for w in text if not w in stops]
                text = " ".join(text)
        return text.strip()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(self.column) > 0:
            X[self.column] = X[self.column].map(self.clean_data)
        else:
            X = X.map(self.clean_data)
        return X
