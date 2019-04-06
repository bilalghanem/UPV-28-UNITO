import warnings
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from os.path import join, exists
import gensim

# config
np.random.seed(0)

class cosine_parent_source_w2v(BaseEstimator, TransformerMixin):
    def __init__(self, path='', n_jobs=1, embeddings_path=''):
        self.path = path
        self.n_jobs = n_jobs
        self.train_size = 0
        self.embeddings_matrix = {}
        self.embeddings_path = embeddings_path

    def build_w2v_sentence(self, X):
        X['text'] = X['text'].map(lambda x: str(x).replace('usermention', '').replace('hashtag', '').replace('http', '').split())
        X_embedd = np.stack(np.mean(np.stack([self.embeddings_matrix[word] if word in self.embeddings_matrix.vocab else np.zeros(shape=(300, )) for word in sent]), axis=0)
                                if len(sent) > 0 else np.zeros(shape=(300, )) for sent in X['text'].tolist())
        return X_embedd

    def fit(self, X, y=None):
        self.train_size = len(X)
        return self

    def load_pretrained_embeddings(self, PATH):
        print("Loading Embeddings Model .. ")
        if PATH.__contains__('.bin'):
            embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=True)
            # embeddings_index.save_word2vec_format('D:/GoogleNews-vectors-negative300.txt', binary=False)
        else:
            def get_coefs(word, *arr):
                return word, np.asarray(arr, dtype='float32')

            if str(PATH).__contains__('wiki'): # 'wiki-news-300d-1M.vec'
                embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(PATH) if len(o) > 100)
            else:
                embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(PATH, encoding='latin'))
        return embeddings_index

    def transform(self, X):
        feats = []
        X = X.reset_index(drop=True)
        if len(X) == self.train_size:
            data_type = 'train'
        else:
            data_type = 'test'


        file_name = join(self.path, 'preprocessed_data/cosine_w2v_{}_.npy'.format(data_type))
        if exists(file_name):
            feats = np.load(file_name)
        else:
            self.embeddings_matrix = self.load_pretrained_embeddings(self.embeddings_path)

            X_embedd = self.build_w2v_sentence(X)
            loop = tqdm(X.iterrows())
            loop.set_description('Cosine_w2v({})'.format(data_type))
            for i, row in loop:
                source = X[(X['event'] == row['event']) & (X['id'] == row['source_id'])]
                if len(source) > 0:
                    if i != source.index[0]:
                        source_cosine = cosine_similarity(X_embedd[i, :].reshape(1,-1), X_embedd[source.index[0], :].reshape(1,-1))[0][0]
                    else:
                        source_cosine = -1
                else:
                    raise ValueError('NO ELSE FOR SOURCE')
                parent = X[(X['event'] == row['event']) & (X['id'] == row['parent'])]
                if len(parent) > 0:
                    parent_cosine = cosine_similarity(X_embedd[i, :].reshape(1, -1), X_embedd[parent.index[0], :].reshape(1,-1))[0][0]
                else:
                    parent_cosine = -1
                similarities = [source_cosine, parent_cosine]
                feats.append(similarities)
            feats = np.asarray(feats)
            np.save(file_name, feats)

            self.embeddings_matrix = None

        return feats

class avg_parent_source_w2v(BaseEstimator, TransformerMixin):
    def __init__(self, path='', n_jobs=1, embeddings_path=''):
        self.path = path
        self.n_jobs = n_jobs
        self.train_size = 0
        self.avg_clusters = dict()
        self.embeddings_matrix = {}
        self.embeddings_path = embeddings_path

    def build_w2v_sentence(self, X):
        X['text'] = X['text'].map(lambda x: str(x).replace('usermention', '').replace('hashtag', '').replace('http', '').split())
        X_embedd = np.stack(np.mean(np.stack([self.embeddings_matrix[word] if word in self.embeddings_matrix.vocab else np.zeros(shape=(300, )) for word in sent]), axis=0)
                                if len(sent) > 0 else np.zeros(shape=(300, )) for sent in X['text'].tolist())
        return X_embedd

    def avg_vectors_of_labels(self, row_feat_vector, level):
        if not level in self.avg_clusters:
            level = list(self.avg_clusters.keys())[-1]
        similarities = []
        levels = [0, level]
        for level in levels:
            for label in self.avg_clusters[level]:
                 similarities.append(cosine_similarity(row_feat_vector, self.avg_clusters[level][label])[0][0])
        return similarities

    def load_pretrained_embeddings(self, PATH):
        print("Loading Embeddings Model .. ")
        if PATH.__contains__('.bin'):
            embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=True)
            # embeddings_index.save_word2vec_format('D:/GoogleNews-vectors-negative300.txt', binary=False)
        else:
            def get_coefs(word, *arr):
                return word, np.asarray(arr, dtype='float32')

            if str(PATH).__contains__('wiki'): # 'wiki-news-300d-1M.vec'
                embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(PATH) if len(o) > 100)
            else:
                embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(PATH, encoding='latin'))
        return embeddings_index

    def fit(self, X, y=None):
        self.train_size = len(X)
        if len(X) == self.train_size:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/avg_w2v_{}_.npy'.format(data_type))
        if not exists(file_name):
            self.embeddings_matrix = self.load_pretrained_embeddings(self.embeddings_path)
            X = X.reset_index(drop=True)
            self.X_embedd = self.build_w2v_sentence(X)
            labels = set(y.tolist())
            levels = {0: np.arange(1, 30),
                      1: [1],
                      2: [2],
                      3: [3],
                      4: [4],
                      5: np.arange(5, 30)
                      }
            for level, level_value in levels.items():
                level_avg = {}
                for label in labels:
                    texts = X[(X['label'] == label) & (X['level'].isin(level_value))]
                    label_vectors = self.X_embedd[texts.index.values].astype(np.float)
                    avg = np.mean(label_vectors, axis=0)
                    where_are_NaNs = np.isnan(avg)
                    avg[where_are_NaNs] = 0
                    level_avg[label] = avg.reshape(1, -1)
                self.avg_clusters[level] = level_avg
            self.embeddings_matrix = None
        return self

    def transform(self, X):
        feats = []
        X = X.reset_index(drop=True)
        if len(X) == self.train_size:
            data_type = 'train'
        else:
            data_type = 'test'
        file_name = join(self.path, 'preprocessed_data/avg_w2v_{}_.npy'.format(data_type))
        if exists(file_name):
            feats = np.load(file_name)
        else:
            self.embeddings_matrix = gensim.models.KeyedVectors.load_word2vec_format(self.embeddings_path, binary=True)

            X_embedd = self.build_w2v_sentence(X)
            loop = tqdm(X.iterrows())
            loop.set_description('Avg_w2v ({})'.format(data_type))
            for i, row in loop:
                similarities = []
                similarities.extend(self.avg_vectors_of_labels(X_embedd[i, :].reshape(1,-1), row['level']))
                feats.append(similarities)
            feats = np.asarray(feats)
            np.save(file_name, feats)

            self.embeddings_matrix = None
        return feats



if __name__ == '__main__':
    s = ["I don't want to be sad"]
    res = cosine_parent_source_w2v().fit_transform(s)
    print(res)
