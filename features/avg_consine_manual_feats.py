import warnings
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from os.path import join, exists
import os

# config
np.random.seed(0)

class cosine_parent_source_feats(BaseEstimator, TransformerMixin):
    def __init__(self, path='', n_jobs=1):
        self.path = path
        self.n_jobs = n_jobs
        self.train_size = 0
        self.Standard_scalar = StandardScaler()

    def load_manual_feats(self, data_type, data_size=0):
        feats = np.zeros(shape=[data_size, 1])
        path = join(self.path, 'preprocessed_data')
        for filename in os.listdir(path):
            if filename.endswith("{}.npy".format(data_type)):
                feats = np.column_stack((feats, np.load(join(path, filename))))

        feats = np.delete(feats, 0, 1) # sec. parm for pos, third: 0=row, 1=column
        return feats

    def fit(self, X, y=None):
        self.train_size = len(X)

        X = X.reset_index(drop=True)
        if len(X) == self.train_size:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/cosine_feats_{}_.npy'.format(data_type))
        if not exists(file_name):
            X = X.reset_index(drop=True)
            self.manual_feats = self.load_manual_feats('train', data_size=len(X))
            self.Standard_scalar.fit(self.manual_feats)
        return self

    def transform(self, X):
        feats = []
        X = X.reset_index(drop=True)
        if len(X) == self.train_size:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/cosine_feats_{}_.npy'.format(data_type))
        if exists(file_name):
            feats = np.load(file_name)
        else:
            ############## ================================================== for the cosine with parent and source
            manual_feats = self.load_manual_feats(data_type, data_size=len(X))
            manual_feats = self.Standard_scalar.transform(manual_feats)
            loop = tqdm(X.iterrows())
            loop.set_description('Cosine_feats ({})'.format(data_type))

            for i, row in loop:
                source = X[(X['event'] == row['event']) & (X['id'] == row['source_id'])]
                if len(source) > 0:
                    if i != source.index[0]:
                        source_cosine = cosine_similarity(manual_feats[i, :].reshape(1,-1), manual_feats[source.index[0], :].reshape(1,-1))[0][0]
                    else:
                        source_cosine = -1
                    # if i == 5:
                    #     np.save('reply.npy', manual_feats[i, :].reshape(1,-1))
                    #     np.save('source.npy', manual_feats[source.index[0], :].reshape(1,-1))
                else:
                    raise ValueError('NO ELSE FOR SOURCE')
                parent = X[(X['event'] == row['event']) & (X['id'] == row['parent'])]
                if len(parent) > 0:
                    parent_cosine = cosine_similarity(manual_feats[i, :].reshape(1, -1), manual_feats[parent.index[0], :].reshape(1,-1))[0][0]
                else:
                    parent_cosine = -1

                similarities = [source_cosine, parent_cosine]
                feats.append(similarities)
            feats = np.asarray(feats)
            np.save(file_name, feats)

        return feats

class avg_parent_source_feats(BaseEstimator, TransformerMixin):
    def __init__(self, path='', n_jobs=1):
        self.path = path
        self.n_jobs = n_jobs
        self.train_size = 0
        self.avg_clusters = dict()
        self.Standard_scalar = StandardScaler()

    def load_manual_feats(self, data_type, data_size=0):
        feats = np.zeros(shape=[data_size, 1])
        path = join(self.path, 'preprocessed_data')
        for filename in os.listdir(path):
            if filename.endswith("{}.npy".format(data_type)):
                feats = np.column_stack((feats, np.load(join(path, filename))))

        feats = np.delete(feats, 0, 1) # sec. parm for pos, third: 0=row, 1=column
        return feats

    def avg_vectors_of_labels(self, row_feat_vector, level):
        if not level in self.avg_clusters:
            level = list(self.avg_clusters.keys())[-1]
        similarities = []
        levels = [0, level]
        for level in levels:
            for label in self.avg_clusters[level]:
                 similarities.append(cosine_similarity(row_feat_vector, self.avg_clusters[level][label])[0][0])
        return similarities

    def fit(self, X, y=None):
        self.train_size = len(X)

        X = X.reset_index(drop=True)
        if len(X) == self.train_size:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/avg_feats_{}_.npy'.format(data_type))
        if not exists(file_name):
            X = X.reset_index(drop=True)
            self.manual_feats = self.load_manual_feats('train', data_size=len(X))
            self.manual_feats = self.Standard_scalar.fit_transform(self.manual_feats)

            labels = set(y.tolist())
            levels = {0: np.arange(1, 30),
                      1: [1],
                      2: [2],
                      3: [3],
                      4: [4],
                      5: np.arange(5, 30)
                      }
            for level, level_value in levels.items():
                # print('--------- level {} ----------'.format(level))
                level_avg = {}
                for label in labels:
                    texts = X[(X['label'] == label) & (X['level'].isin(level_value))]
                    label_vectors = self.manual_feats[texts.index.values].astype(np.float)
                    avg = np.mean(label_vectors, axis=0)
                    where_are_NaNs = np.isnan(avg)
                    avg[where_are_NaNs] = 0
                    level_avg[label] = avg.reshape(1, -1)
                    # print('label: {} : {}'.format(label, len(texts)))
                self.avg_clusters[level] = level_avg
        return self

    def transform(self, X):
        feats = []
        X = X.reset_index(drop=True)
        if len(X) == self.train_size:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/avg_feats_{}_.npy'.format(data_type))
        if exists(file_name):
            feats = np.load(file_name)
        else:
            ############## ================================================== for the cosine with parent and source
            manual_feats = self.load_manual_feats(data_type, data_size=len(X))
            manual_feats = self.Standard_scalar.transform(manual_feats)
            loop = tqdm(X.iterrows())
            loop.set_description('Avg_feats ({})'.format(data_type))

            for i, row in loop:
                similarities = []
                similarities.extend(self.avg_vectors_of_labels(manual_feats[i, :].reshape(1,-1), row['level']))
                feats.append(similarities)
            feats = np.asarray(feats)
            np.save(file_name, feats)

        return feats



if __name__ == '__main__':
    s = ["I don't want to be sad"]
    res = cosine_parent_source_feats().fit_transform(s)
    print(res)