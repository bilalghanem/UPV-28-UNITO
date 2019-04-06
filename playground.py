import datetime as dt
from tqdm import tqdm
import time, json
import ast
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import ufal.udpipe, conllu, re, sys

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X[[self.column]]

class Length(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = [len(x) for x in X]
        return X

class get_parent_childs():
    def __init__(self, tree=dict(), target=""):
        self.parent = 'source'
        self.child = 0
        self.get(tree, target)

    def get(self, struct_dict, target):
        for item in struct_dict:
            childs = struct_dict[item]
            if item == target:
                self.child = len(childs)
                return 1
            if len(childs):
                res = self.get(childs, target)
                if res == 1:
                    self.parent = item

class Model_udpipe:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()
        return output

def prepare_submission():
    with open('learning/answer.json') as data:
        answer = json.load(data)
    print(len(answer['subtaskaenglish']))

    with open('learning/twitter.json') as data:
        twitter = json.load(data)

    with open('learning/reddit.json') as data:
        reddit = json.load(data)

    print(len(twitter['subtaskaenglish']))

    answer['subtaskaenglish'].update(twitter['subtaskaenglish'])
    answer['subtaskaenglish'].update(reddit['subtaskaenglish'])

    print(len(answer['subtaskaenglish']))

    with open('learning/edited_answer.json', 'w') as outfile:
        json.dump(answer, outfile)

    print()

if __name__ == "__main__":
    scaler_twitter = MinMaxScaler(feature_range=(0, 1), copy=False)
    x = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
    x2 = np.array([4,5,6,6]).reshape(-1,1)
    x3 = np.array([18]).reshape(-1,1)
    scaler_twitter.fit(x)
    print(scaler_twitter.transform(x3).tolist()[0][0])
    print(type(scaler_twitter.transform(x3).tolist()[0][0]))
